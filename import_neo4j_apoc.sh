#!/usr/bin/env bash
set -euo pipefail

# Config
NEO4J_USER="neo4j"
NEO4J_PASS="neo4jtest12"
DB_NAME="wiki.db"
SERVICE="neo4j"
JSONL="file:///wiki_entities_neo4j.jsonl"

wait_for_neo4j() {
  echo "Waiting for Neo4j to accept connections..."
  for i in {1..60}; do
    if docker compose exec -T "$SERVICE" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" -d "$DB_NAME" "RETURN 1" >/dev/null 2>&1; then
      echo "Neo4j is ready."
      return 0
    fi
    sleep 2
  done
  echo "Neo4j did not become ready in time." >&2
  exit 1
}

run_cypher() {
  local query="$1"
  docker compose exec -T "$SERVICE" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" -d "$DB_NAME" "$query"
}

# Ensure container is up
if ! docker compose ps "$SERVICE" >/dev/null 2>&1; then
  echo "Starting Neo4j via docker compose..."
  docker compose up -d "$SERVICE"
fi

wait_for_neo4j

echo "Wiping database (nodes, relationships, and existing constraint) to start fresh..."
run_cypher "DROP CONSTRAINT page_id IF EXISTS;"
run_cypher "MATCH (n) DETACH DELETE n;"

echo "Creating constraint..."
run_cypher "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (n:Page) REQUIRE n.id IS UNIQUE;"

echo "Importing nodes from $JSONL via APOC with batching (periodic iterate)..."
run_cypher "CALL apoc.periodic.iterate('CALL apoc.load.json(\"$JSONL\") YIELD value RETURN value','MERGE (n:Page {id: toInteger(value.id)}) SET n.title = value.title, n.category = value.category, n.wikidata_id = value.wikidata_id, n.wikipedia = apoc.convert.toJson(value.wikipedia), n.wikidata = apoc.convert.toJson(value.wikidata), n.start_time = value.start_time, n.end_time = value.end_time, n.point_in_time = value.point_in_time, n.date_of_birth = value.date_of_birth, n.date_of_death = value.date_of_death, n.country = apoc.convert.toList(coalesce(value.country, value.country_codes)), n.country_codes = apoc.convert.toList(value.country_codes), n.gender = coalesce(value.gender, value.gender_code), n.gender_code = value.gender_code, n.occupation = apoc.convert.toList(coalesce(value.occupation, value.occupation_codes)), n.occupation_codes = apoc.convert.toList(value.occupation_codes), n.instance_of = apoc.convert.toList(coalesce(value.instance_of, value.instance_of_codes)), n.instance_of_codes = apoc.convert.toList(value.instance_of_codes), n.subclass_of = apoc.convert.toList(coalesce(value.subclass_of, value.subclass_of_codes)), n.subclass_of_codes = apoc.convert.toList(value.subclass_of_codes), n.location = apoc.convert.toList(coalesce(value.location, value.location_codes)), n.location_codes = apoc.convert.toList(value.location_codes), n.latitude = value.latitude, n.longitude = value.longitude', {batchSize: 200, parallel: true, retries: 1});"

echo "Setting labels..."
run_cypher "MATCH (n:Page) WHERE n.category = 'person' SET n:Person;"
run_cypher "MATCH (n:Page) WHERE n.category = 'event'  SET n:Event;"

echo "Removing legacy directional edges (LINKS_TO) and previous undirected edges (BICONNECTS)..."
run_cypher "MATCH ()-[r:LINKS_TO]->() DELETE r;"
run_cypher "MATCH ()-[r:BICONNECTS]-() DELETE r;"

echo "Importing undirected edges from $JSONL via APOC with batching (canonical pair, weight=1)..."
run_cypher "CALL apoc.periodic.iterate('CALL apoc.load.json(\"$JSONL\") YIELD value RETURN value','WITH toInteger(value.id) AS src, apoc.coll.toSet(coalesce(value.links, [])) AS links UNWIND links AS tgtInt WITH src, toInteger(tgtInt) AS tgt WHERE src IS NOT NULL AND tgt IS NOT NULL AND src <> tgt WITH apoc.coll.toSet(CASE WHEN src <= tgt THEN [src, tgt] ELSE [tgt, src] END) AS pair WITH pair[0] AS lo, pair[1] AS hi MATCH (a:Page {id: lo}), (b:Page {id: hi}) MERGE (a)-[r:BICONNECTS]->(b) ON CREATE SET r.weight = 1 ON MATCH SET r.weight = 1', {batchSize: 500, parallel: false, retries: 1});"

echo "Done. Try visualization queries in Neo4j Browser, e.g.:"
echo "  MATCH (p:Person)-[r:BICONNECTS]-(e:Event) RETURN p,r,e LIMIT 100;"

echo "\nBasic statistics:"
echo "- Total pages:"
run_cypher "MATCH (n:Page) RETURN count(n) AS pages;"
echo "- Persons / Events:"
run_cypher "MATCH (n:Person) RETURN count(n) AS persons;"
run_cypher "MATCH (n:Event)  RETURN count(n) AS events;"
echo "- BICONNECTS edges:"
run_cypher "MATCH ()-[r:BICONNECTS]-() RETURN count(r) AS edges;"
echo "- Average degree (undirected):"
run_cypher "MATCH (n:Page) RETURN avg(count { (n)-[:BICONNECTS]-() }) AS avg_degree;"
echo "- Persons with gender / country:"
run_cypher "MATCH (n:Person) WHERE n.gender IS NOT NULL  RETURN count(n) AS with_gender;"
run_cypher "MATCH (n:Person) WHERE n.country IS NOT NULL RETURN count(n) AS with_country;"
