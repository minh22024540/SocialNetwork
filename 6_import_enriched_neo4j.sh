#!/usr/bin/env bash
set -euo pipefail

# Config
NEO4J_USER="neo4j"
NEO4J_PASS="neo4jtest12"
DB_NAME="wiki.db"  # Use the existing database
CONTAINER_NAME="neo4j"
JSONL="file:///import/wiki_entities_with_weights.jsonl"
ENRICHED_JSONL="file:///import/enriched_relationships.jsonl"

wait_for_neo4j() {
  echo "Waiting for Neo4j to accept connections..."
  for i in {1..60}; do
    if docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" "RETURN 1" >/dev/null 2>&1; then
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
  local db="${2:-}"
  if [ -n "$db" ]; then
    docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" -d "$db" "$query"
  else
    docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" "$query"
  fi
}

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
  echo "Error: Neo4j container '$CONTAINER_NAME' is not running."
  echo "Please start it with: docker start $CONTAINER_NAME"
  exit 1
fi

wait_for_neo4j

echo "Using database: $DB_NAME"
echo "Note: We'll clear only enriched data (nodes with HAS_RELATIONSHIP edges) to preserve original data"
echo "Wiping enriched nodes and relationships to start fresh..."
run_cypher "DROP CONSTRAINT page_id IF EXISTS;" "$DB_NAME" || true
# Only delete nodes that have enriched relationships, or all if starting fresh
run_cypher "MATCH (n) WHERE EXISTS { (n)-[:HAS_RELATIONSHIP]-() } DETACH DELETE n;" "$DB_NAME" || run_cypher "MATCH (n) DETACH DELETE n;" "$DB_NAME"

echo "Creating constraint..."
run_cypher "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (n:Page) REQUIRE n.id IS UNIQUE;" "$DB_NAME"

echo "Importing nodes from $JSONL via APOC with batching (periodic iterate)..."
run_cypher "CALL apoc.periodic.iterate('CALL apoc.load.json(\"$JSONL\") YIELD value RETURN value','WITH value, toInteger(value.wikipedia.id) AS entity_id MERGE (n:Page {id: entity_id}) SET n.title = value.wikipedia.title, n.category = value.category, n.wikidata_id = value.wikidata.id, n.start_time = value.start_time, n.end_time = value.end_time, n.point_in_time = value.point_in_time, n.date_of_birth = value.date_of_birth, n.date_of_death = value.date_of_death, n.country = apoc.convert.toList(coalesce(value.country, value.country_codes)), n.country_codes = apoc.convert.toList(value.country_codes), n.gender = coalesce(value.gender, value.gender_code), n.gender_code = value.gender_code, n.occupation = apoc.convert.toList(coalesce(value.occupation, value.occupation_codes)), n.occupation_codes = apoc.convert.toList(value.occupation_codes), n.instance_of = apoc.convert.toList(coalesce(value.instance_of, value.instance_of_codes)), n.instance_of_codes = apoc.convert.toList(value.instance_of_codes), n.subclass_of = apoc.convert.toList(coalesce(value.subclass_of, value.subclass_of_codes)), n.subclass_of_codes = apoc.convert.toList(value.subclass_of_codes), n.location = apoc.convert.toList(coalesce(value.location, value.location_codes)), n.location_codes = apoc.convert.toList(value.location_codes), n.latitude = value.latitude, n.longitude = value.longitude', {batchSize: 200, parallel: true, retries: 1});" "$DB_NAME"

echo "Setting labels..."
run_cypher "MATCH (n:Page) WHERE n.category = 'person' SET n:Person;" "$DB_NAME"
run_cypher "MATCH (n:Page) WHERE n.category = 'event'  SET n:Event;" "$DB_NAME"

echo "Importing undirected BICONNECTS edges with weights from $JSONL..."
# Handle both integer links and dict links with weights
run_cypher "CALL apoc.periodic.iterate('CALL apoc.load.json(\"$JSONL\") YIELD value RETURN value','WITH value, toInteger(value.wikipedia.id) AS src, value.links AS links UNWIND links AS link_item WITH src, CASE WHEN link_item IS NULL THEN NULL WHEN link_item IS INTEGER THEN toInteger(link_item) WHEN link_item.target_id IS NOT NULL THEN toInteger(link_item.target_id) ELSE toInteger(link_item) END AS tgt, CASE WHEN link_item IS INTEGER THEN 1.0 WHEN link_item.weight IS NOT NULL THEN toFloat(link_item.weight) ELSE 1.0 END AS weight WHERE src IS NOT NULL AND tgt IS NOT NULL AND src <> tgt WITH CASE WHEN src <= tgt THEN [src, tgt] ELSE [tgt, src] END AS pair, weight WITH pair[0] AS lo, pair[1] AS hi, weight MATCH (a:Page {id: lo}), (b:Page {id: hi}) MERGE (a)-[r:BICONNECTS]-(b) ON CREATE SET r.weight = weight ON MATCH SET r.weight = weight', {batchSize: 500, parallel: false, retries: 1});" "$DB_NAME"

echo "Importing enriched relationships (HAS_RELATIONSHIP) from $ENRICHED_JSONL..."
run_cypher "CALL apoc.periodic.iterate('CALL apoc.load.json(\"$ENRICHED_JSONL\") YIELD value RETURN value','MATCH (source:Page {id: toInteger(value.source_id)}), (target:Page {id: toInteger(value.target_id)}) MERGE (source)-[r:HAS_RELATIONSHIP]->(target) SET r.type = value.relationship_type, r.confidence = toFloat(value.confidence), r.evidence_text = value.evidence_text, r.source = value.source', {batchSize: 500, parallel: false, retries: 1});" "$DB_NAME"

echo "Done. Enriched graph imported to database: $DB_NAME"
echo ""
echo "Try visualization queries in Neo4j Browser, e.g.:"
echo "  MATCH (p:Person)-[r:HAS_RELATIONSHIP]->(e:Event) RETURN p,r,e LIMIT 100;"
echo "  MATCH (p:Person)-[r:BICONNECTS]-(e:Event) RETURN p,r,e LIMIT 100;"

echo ""
echo "Basic statistics:"
echo "- Total pages:"
run_cypher "MATCH (n:Page) RETURN count(n) AS pages;" "$DB_NAME"
echo "- Persons / Events:"
run_cypher "MATCH (n:Person) RETURN count(n) AS persons;" "$DB_NAME"
run_cypher "MATCH (n:Event)  RETURN count(n) AS events;" "$DB_NAME"
echo "- BICONNECTS edges:"
run_cypher "MATCH ()-[r:BICONNECTS]-() RETURN count(r) AS edges;" "$DB_NAME"
echo "- HAS_RELATIONSHIP edges:"
run_cypher "MATCH ()-[r:HAS_RELATIONSHIP]->() RETURN count(r) AS edges;" "$DB_NAME"
echo "- Relationship type distribution:"
run_cypher "MATCH ()-[r:HAS_RELATIONSHIP]->() RETURN r.type AS type, count(*) AS count ORDER BY count DESC;" "$DB_NAME"

