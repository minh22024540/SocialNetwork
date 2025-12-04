"""
Sample multi-hop paths from Neo4j graph for question generation.

This module extracts diverse paths of different hop lengths (2, 3, 4) from the
social network graph, ensuring variety in relationship types, persons, and events.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import random

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j package not installed. Install with: pip install neo4j")
    raise


# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4jtest12"
DB_NAME = "wiki.db"  # Database name as shown in SHOW DATABASES


def get_neo4j_driver():
    """Get Neo4j driver instance.

    Returns:
        GraphDatabase driver instance configured with connection parameters.
    """
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def run_cypher_query(
    driver, query: str, database: str = DB_NAME
) -> List[Dict]:
    """Execute a Cypher query and return results as list of dictionaries.

    Args:
        driver: Neo4j driver instance.
        query: Cypher query string.
        database: Database name. Defaults to DB_NAME.

    Returns:
        List of dictionaries, each representing a row of results.
        Returns empty list on error.
    """
    try:
        with driver.session(database=database) as session:
            result = session.run(query)
            rows = []
            for record in result:
                row = {}
                for key in record.keys():
                    value = record[key]
                    # Convert Neo4j types to Python types
                    if hasattr(value, '__dict__'):
                        value = str(value)
                    row[key] = value
                rows.append(row)
            return rows
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
        return []


def sample_2hop_paths(driver, count: int = 1000) -> List[Dict]:
    """Sample 2-hop paths: Person → Event → Person.

    Args:
        driver: Neo4j driver instance.
        count: Number of paths to sample. Defaults to 1000.

    Returns:
        List of path dictionaries, each containing:
        - hop_count: 2
        - path: List of nodes [Person, Event, Person]
        - relationships: List of relationship dicts with type, confidence, evidence_text
    """
    print(f"Sampling {count} 2-hop paths...")
    
    # Query to get 2-hop paths
    query = """
    MATCH path = (p1:Person)-[r1:HAS_RELATIONSHIP]->(e:Event)<-[r2:HAS_RELATIONSHIP]-(p2:Person)
    WHERE p1 <> p2
    RETURN p1.id AS person1_id, p1.title AS person1_name,
           e.id AS event_id, e.title AS event_name,
           p2.id AS person2_id, p2.title AS person2_name,
           r1.type AS rel1_type, r2.type AS rel2_type,
           r1.confidence AS rel1_confidence, r2.confidence AS rel2_confidence,
           r1.evidence_text AS rel1_evidence, r2.evidence_text AS rel2_evidence
    LIMIT 10000
    """
    
    all_paths = run_cypher_query(driver, query)
    
    if len(all_paths) < count:
        print(f"Warning: Only found {len(all_paths)} 2-hop paths, requested {count}")
        return all_paths
    
    # Sample diverse paths
    sampled = random.sample(all_paths, min(count, len(all_paths)))
    
    # Format paths
    formatted_paths = []
    for path in sampled:
        formatted_paths.append({
            "hop_count": 2,
            "path": [
                {"type": "person", "id": path["person1_id"], "name": path["person1_name"]},
                {"type": "event", "id": path["event_id"], "name": path["event_name"]},
                {"type": "person", "id": path["person2_id"], "name": path["person2_name"]}
            ],
            "relationships": [
                {
                    "type": path["rel1_type"],
                    "confidence": path.get("rel1_confidence", 0.5),
                    "evidence_text": path.get("rel1_evidence", "")
                },
                {
                    "type": path["rel2_type"],
                    "confidence": path.get("rel2_confidence", 0.5),
                    "evidence_text": path.get("rel2_evidence", "")
                }
            ]
        })
    
    print(f"Sampled {len(formatted_paths)} 2-hop paths")
    return formatted_paths


def sample_3hop_paths(driver, count: int = 700) -> List[Dict]:
    """Sample 3-hop paths: Person → Event → Person → Event → Person.

    Args:
        driver: Neo4j driver instance.
        count: Number of paths to sample. Defaults to 700.

    Returns:
        List of path dictionaries, each containing:
        - hop_count: 3
        - path: List of nodes [Person, Event, Person, Event, Person]
        - relationships: List of relationship dicts with type, confidence, evidence_text
    """
    print(f"Sampling {count} 3-hop paths...")
    
    # Query to get 3-hop paths
    query = """
    MATCH path = (p1:Person)-[r1:HAS_RELATIONSHIP]->(e1:Event)<-[r2:HAS_RELATIONSHIP]-(p2:Person)-[r3:HAS_RELATIONSHIP]->(e2:Event)<-[r4:HAS_RELATIONSHIP]-(p3:Person)
    WHERE p1 <> p2 AND p2 <> p3 AND p1 <> p3
    RETURN p1.id AS person1_id, p1.title AS person1_name,
           e1.id AS event1_id, e1.title AS event1_name,
           p2.id AS person2_id, p2.title AS person2_name,
           e2.id AS event2_id, e2.title AS event2_name,
           p3.id AS person3_id, p3.title AS person3_name,
           r1.type AS rel1_type, r2.type AS rel2_type,
           r3.type AS rel3_type, r4.type AS rel4_type,
           r1.confidence AS rel1_confidence, r2.confidence AS rel2_confidence,
           r3.confidence AS rel3_confidence, r4.confidence AS rel4_confidence,
           r1.evidence_text AS rel1_evidence, r2.evidence_text AS rel2_evidence,
           r3.evidence_text AS rel3_evidence, r4.evidence_text AS rel4_evidence
    LIMIT 10000
    """
    
    all_paths = run_cypher_query(driver, query)
    
    if len(all_paths) < count:
        print(f"Warning: Only found {len(all_paths)} 3-hop paths, requested {count}")
        return all_paths
    
    # Sample diverse paths
    sampled = random.sample(all_paths, min(count, len(all_paths)))
    
    # Format paths
    formatted_paths = []
    for path in sampled:
        formatted_paths.append({
            "hop_count": 3,
            "path": [
                {"type": "person", "id": path["person1_id"], "name": path["person1_name"]},
                {"type": "event", "id": path["event1_id"], "name": path["event1_name"]},
                {"type": "person", "id": path["person2_id"], "name": path["person2_name"]},
                {"type": "event", "id": path["event2_id"], "name": path["event2_name"]},
                {"type": "person", "id": path["person3_id"], "name": path["person3_name"]}
            ],
            "relationships": [
                {
                    "type": path["rel1_type"],
                    "confidence": path.get("rel1_confidence", 0.5),
                    "evidence_text": path.get("rel1_evidence", "")
                },
                {
                    "type": path["rel2_type"],
                    "confidence": path.get("rel2_confidence", 0.5),
                    "evidence_text": path.get("rel2_evidence", "")
                },
                {
                    "type": path["rel3_type"],
                    "confidence": path.get("rel3_confidence", 0.5),
                    "evidence_text": path.get("rel3_evidence", "")
                },
                {
                    "type": path["rel4_type"],
                    "confidence": path.get("rel4_confidence", 0.5),
                    "evidence_text": path.get("rel4_evidence", "")
                }
            ]
        })
    
    print(f"Sampled {len(formatted_paths)} 3-hop paths")
    return formatted_paths


def sample_4hop_paths(driver, count: int = 300) -> List[Dict]:
    """Sample 4-hop paths: Person → Event → Person → Event → Person → Event → Person.

    Args:
        driver: Neo4j driver instance.
        count: Number of paths to sample. Defaults to 300.

    Returns:
        List of path dictionaries, each containing:
        - hop_count: 4
        - path: List of nodes [Person, Event, Person, Event, Person, Event, Person]
        - relationships: List of relationship dicts with type, confidence, evidence_text
        Returns empty list if insufficient paths found in database.
    """
    print(f"Sampling {count} 4-hop paths...")
    
    # Query to get 4-hop paths
    query = """
    MATCH path = (p1:Person)-[r1:HAS_RELATIONSHIP]->(e1:Event)<-[r2:HAS_RELATIONSHIP]-(p2:Person)
          -[r3:HAS_RELATIONSHIP]->(e2:Event)<-[r4:HAS_RELATIONSHIP]-(p3:Person)
          -[r5:HAS_RELATIONSHIP]->(e3:Event)<-[r6:HAS_RELATIONSHIP]-(p4:Person)
    WHERE p1 <> p2 AND p2 <> p3 AND p3 <> p4 
          AND p1 <> p3 AND p1 <> p4 AND p2 <> p4
    RETURN p1.id AS person1_id, p1.title AS person1_name,
           e1.id AS event1_id, e1.title AS event1_name,
           p2.id AS person2_id, p2.title AS person2_name,
           e2.id AS event2_id, e2.title AS event2_name,
           p3.id AS person3_id, p3.title AS person3_name,
           e3.id AS event3_id, e3.title AS event3_name,
           p4.id AS person4_id, p4.title AS person4_name,
           r1.type AS rel1_type, r2.type AS rel2_type,
           r3.type AS rel3_type, r4.type AS rel4_type,
           r5.type AS rel5_type, r6.type AS rel6_type,
           r1.confidence AS rel1_confidence, r2.confidence AS rel2_confidence,
           r3.confidence AS rel3_confidence, r4.confidence AS rel4_confidence,
           r5.confidence AS rel5_confidence, r6.confidence AS rel6_confidence,
           r1.evidence_text AS rel1_evidence, r2.evidence_text AS rel2_evidence,
           r3.evidence_text AS rel3_evidence, r4.evidence_text AS rel4_evidence,
           r5.evidence_text AS rel5_evidence, r6.evidence_text AS rel6_evidence
    LIMIT 10000
    """
    
    all_paths = run_cypher_query(driver, query)
    
    if len(all_paths) < count:
        print(f"Warning: Only found {len(all_paths)} 4-hop paths, requested {count}")
        print("Will adjust distribution to use more 2-hop and 3-hop paths")
        return []
    
    # Sample diverse paths
    sampled = random.sample(all_paths, min(count, len(all_paths)))
    
    # Format paths
    formatted_paths = []
    for path in sampled:
        formatted_paths.append({
            "hop_count": 4,
            "path": [
                {"type": "person", "id": path["person1_id"], "name": path["person1_name"]},
                {"type": "event", "id": path["event1_id"], "name": path["event1_name"]},
                {"type": "person", "id": path["person2_id"], "name": path["person2_name"]},
                {"type": "event", "id": path["event2_id"], "name": path["event2_name"]},
                {"type": "person", "id": path["person3_id"], "name": path["person3_name"]},
                {"type": "event", "id": path["event3_id"], "name": path["event3_name"]},
                {"type": "person", "id": path["person4_id"], "name": path["person4_name"]}
            ],
            "relationships": [
                {
                    "type": path["rel1_type"],
                    "confidence": path.get("rel1_confidence", 0.5),
                    "evidence_text": path.get("rel1_evidence", "")
                },
                {
                    "type": path["rel2_type"],
                    "confidence": path.get("rel2_confidence", 0.5),
                    "evidence_text": path.get("rel2_evidence", "")
                },
                {
                    "type": path["rel3_type"],
                    "confidence": path.get("rel3_confidence", 0.5),
                    "evidence_text": path.get("rel3_evidence", "")
                },
                {
                    "type": path["rel4_type"],
                    "confidence": path.get("rel4_confidence", 0.5),
                    "evidence_text": path.get("rel4_evidence", "")
                },
                {
                    "type": path["rel5_type"],
                    "confidence": path.get("rel5_confidence", 0.5),
                    "evidence_text": path.get("rel5_evidence", "")
                },
                {
                    "type": path["rel6_type"],
                    "confidence": path.get("rel6_confidence", 0.5),
                    "evidence_text": path.get("rel6_evidence", "")
                }
            ]
        })
    
    print(f"Sampled {len(formatted_paths)} 4-hop paths")
    return formatted_paths


def main() -> None:
    """Main function to sample all paths and save to JSON.

    Samples 2000 total paths (1000 2-hop, 700 3-hop, 300 4-hop) from Neo4j
    and saves them to graph_paths_for_questions.json for question generation.
    Uses deterministic random seed (42) to ensure reproducible results.
    """
    print("=" * 80)
    print("Sampling Multi-hop Graph Paths")
    print("=" * 80)
    
    # Set random seed for deterministic sampling
    # This ensures the same paths are sampled every time
    random.seed(42)
    print("Random seed set to 42 for deterministic sampling")
    
    # Connect to Neo4j
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        print("Connected to Neo4j")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("Make sure Neo4j is running and accessible at bolt://localhost:7687")
        return
    
    try:
        # Sample paths
        paths_2hop = sample_2hop_paths(driver, count=1000)
        paths_3hop = sample_3hop_paths(driver, count=700)
        paths_4hop = sample_4hop_paths(driver, count=300)
        
        # Handle fallback if 4-hop is insufficient
        if not paths_4hop:
            print("\n4-hop paths insufficient, adjusting distribution...")
            additional_2hop = 300  # Add 300 more 2-hop to reach 2000 total
            paths_2hop.extend(sample_2hop_paths(driver, count=additional_2hop))
            print(f"New distribution: {len(paths_2hop)} (2-hop), {len(paths_3hop)} (3-hop)")
        
        # Combine all paths
        all_paths = paths_2hop + paths_3hop + paths_4hop
        
        # Save to JSON (relative to SocialNetwork project root)
        output_path = Path(__file__).resolve().parent / "data" / "graph_paths_for_questions.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_paths, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total paths sampled: {len(all_paths)}")
        print(f"  2-hop: {len(paths_2hop)}")
        print(f"  3-hop: {len(paths_3hop)}")
        print(f"  4-hop: {len(paths_4hop)}")
        print(f"\nSaved to: {output_path}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()

