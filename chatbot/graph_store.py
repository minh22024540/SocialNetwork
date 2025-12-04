"""
GraphStore: thin wrapper around Neo4j for GraphRAG.

This is intentionally minimal and explicit (no LangChain) so that the
multi-hop reasoning over the social network graph is easy to inspect
and explain in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase, Driver
except ImportError as e:  # pragma: no cover - runtime import guard
    raise ImportError(
        "neo4j package not installed. Install with: pip install neo4j"
    ) from e


# Keep defaults consistent with the data-loading scripts.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4jtest12"
DB_NAME = "wiki.db"


@dataclass
class GraphNode:
    id: int
    label: str
    title: str


@dataclass
class GraphRelationship:
    start_id: int
    end_id: int
    type: str
    confidence: float
    evidence_text: str


@dataclass
class GraphPath:
    hop_count: int
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]


class GraphStore:
    """Simple Neo4j wrapper focused on the operations we need for GraphRAG."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASS,
        database: str = DB_NAME,
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    # --- low-level helpers -------------------------------------------------

    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Run a Cypher query and return list of dict rows."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            rows: List[Dict[str, Any]] = []
            for record in result:
                row = {}
                for key in record.keys():
                    value = record[key]
                    row[key] = value
                rows.append(row)
            return rows

    # --- basic lookups -----------------------------------------------------

    def get_node_by_id(self, node_id: int) -> Optional[GraphNode]:
        query = """
        MATCH (n)
        WHERE n.id = $node_id
        RETURN labels(n)[0] AS label, n.id AS id, n.title AS title
        LIMIT 1
        """
        rows = self._run(query, {"node_id": node_id})
        if not rows:
            return None
        r = rows[0]
        return GraphNode(id=r["id"], label=r["label"], title=r.get("title", ""))

    def find_nodes_by_title(
        self,
        title_substring: str,
        labels: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """Very simple entity lookup by substring match on `title`."""
        title_substring = title_substring.lower()
        label_filter = ""
        if labels:
            label_filter = " AND ANY(l in labels(n) WHERE l IN $labels)"

        query = f"""
        MATCH (n)
        WHERE toLower(n.title) CONTAINS $q
        {label_filter}
        RETURN labels(n)[0] AS label, n.id AS id, n.title AS title
        LIMIT $limit
        """
        rows = self._run(
            query,
            {"q": title_substring, "labels": labels or [], "limit": int(limit)},
        )
        return [
            GraphNode(id=r["id"], label=r["label"], title=r.get("title", ""))
            for r in rows
        ]

    # --- path / subgraph retrieval -----------------------------------------

    def find_nodes_in_text(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[GraphNode]:
        """Find nodes mentioned in free text using fuzzy matching on title only."""
        text_l = text.lower()
        words = [w for w in text_l.replace("\n", " ").split(" ") if w]

        label_filter = ""
        if labels:
            label_filter = " AND ANY(l IN labels(n) WHERE l IN $labels)"

        query = f"""
        WITH toLower($text) AS full_text, $words AS words
        MATCH (n)
        WHERE
          (
            full_text CONTAINS toLower(n.title)
            OR toLower(n.title) CONTAINS full_text
            OR ANY(w IN words WHERE toLower(n.title) CONTAINS w OR w CONTAINS toLower(n.title))
          )
          {label_filter}
        RETURN labels(n)[0] AS label, n.id AS id, n.title AS title
        LIMIT $limit
        """
        rows = self._run(
            query,
            {
                "text": text_l,
                "words": words,
                "labels": labels or [],
                "limit": int(limit),
            },
        )
        return [
            GraphNode(id=r["id"], label=r["label"], title=r.get("title", ""))
            for r in rows
        ]

    def sample_paths_for_candidates_2hop(
        self,
        candidate_ids: List[int],
        max_paths: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sample 2-hop Person–Event–Person paths touching any of the candidate ids."""
        if not candidate_ids:
            return []

        query = """
        MATCH path = (p1:Person)-[r1:HAS_RELATIONSHIP]->(e:Event)<-[r2:HAS_RELATIONSHIP]-(p2:Person)
        WHERE p1 <> p2
          AND (p1.id IN $ids OR p2.id IN $ids OR e.id IN $ids)
        RETURN p1.id AS person1_id, p1.title AS person1_name,
               e.id AS event_id, e.title AS event_name,
               p2.id AS person2_id, p2.title AS person2_name,
               r1.type AS rel1_type, r2.type AS rel2_type,
               r1.confidence AS rel1_confidence, r2.confidence AS rel2_confidence,
               r1.evidence_text AS rel1_evidence, r2.evidence_text AS rel2_evidence
        LIMIT $limit
        """
        rows = self._run(query, {"ids": candidate_ids, "limit": int(max_paths)})

        paths: List[Dict[str, Any]] = []
        for row in rows:
            paths.append(
                {
                    "hop_count": 2,
                    "path": [
                        {
                            "type": "person",
                            "id": row["person1_id"],
                            "name": row["person1_name"],
                        },
                        {
                            "type": "event",
                            "id": row["event_id"],
                            "name": row["event_name"],
                        },
                        {
                            "type": "person",
                            "id": row["person2_id"],
                            "name": row["person2_name"],
                        },
                    ],
                    "relationships": [
                        {
                            "type": row["rel1_type"],
                            "confidence": row.get("rel1_confidence", 0.5),
                            "evidence_text": row.get("rel1_evidence", ""),
                        },
                        {
                            "type": row["rel2_type"],
                            "confidence": row.get("rel2_confidence", 0.5),
                            "evidence_text": row.get("rel2_evidence", ""),
                        },
                    ],
                }
            )
        return paths

    def sample_paths_for_candidates_3hop(
        self,
        candidate_ids: List[int],
        max_paths: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sample 3-hop Person–Event–Person–Event–Person paths touching candidates."""
        if not candidate_ids:
            return []

        query = """
        MATCH path = (p1:Person)-[r1:HAS_RELATIONSHIP]->(e1:Event)<-[r2:HAS_RELATIONSHIP]-(p2:Person)
              -[r3:HAS_RELATIONSHIP]->(e2:Event)<-[r4:HAS_RELATIONSHIP]-(p3:Person)
        WHERE p1 <> p2 AND p2 <> p3 AND p1 <> p3
          AND (p1.id IN $ids OR p2.id IN $ids OR p3.id IN $ids
               OR e1.id IN $ids OR e2.id IN $ids)
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
        LIMIT $limit
        """
        rows = self._run(query, {"ids": candidate_ids, "limit": int(max_paths)})

        paths: List[Dict[str, Any]] = []
        for row in rows:
            paths.append(
                {
                    "hop_count": 3,
                    "path": [
                        {
                            "type": "person",
                            "id": row["person1_id"],
                            "name": row["person1_name"],
                        },
                        {
                            "type": "event",
                            "id": row["event1_id"],
                            "name": row["event1_name"],
                        },
                        {
                            "type": "person",
                            "id": row["person2_id"],
                            "name": row["person2_name"],
                        },
                        {
                            "type": "event",
                            "id": row["event2_id"],
                            "name": row["event2_name"],
                        },
                        {
                            "type": "person",
                            "id": row["person3_id"],
                            "name": row["person3_name"],
                        },
                    ],
                    "relationships": [
                        {
                            "type": row["rel1_type"],
                            "confidence": row.get("rel1_confidence", 0.5),
                            "evidence_text": row.get("rel1_evidence", ""),
                        },
                        {
                            "type": row["rel2_type"],
                            "confidence": row.get("rel2_confidence", 0.5),
                            "evidence_text": row.get("rel2_evidence", ""),
                        },
                        {
                            "type": row["rel3_type"],
                            "confidence": row.get("rel3_confidence", 0.5),
                            "evidence_text": row.get("rel3_evidence", ""),
                        },
                        {
                            "type": row["rel4_type"],
                            "confidence": row.get("rel4_confidence", 0.5),
                            "evidence_text": row.get("rel4_evidence", ""),
                        },
                    ],
                }
            )
        return paths

    def sample_paths_for_candidates_4hop(
        self,
        candidate_ids: List[int],
        max_paths: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sample 4-hop Person–Event–Person–Event–Person–Event–Person paths."""
        if not candidate_ids:
            return []

        query = """
        MATCH path = (p1:Person)-[r1:HAS_RELATIONSHIP]->(e1:Event)<-[r2:HAS_RELATIONSHIP]-(p2:Person)
              -[r3:HAS_RELATIONSHIP]->(e2:Event)<-[r4:HAS_RELATIONSHIP]-(p3:Person)
              -[r5:HAS_RELATIONSHIP]->(e3:Event)<-[r6:HAS_RELATIONSHIP]-(p4:Person)
        WHERE p1 <> p2 AND p2 <> p3 AND p3 <> p4 AND p1 <> p3 AND p1 <> p4 AND p2 <> p4
          AND (p1.id IN $ids OR p2.id IN $ids OR p3.id IN $ids OR p4.id IN $ids
               OR e1.id IN $ids OR e2.id IN $ids OR e3.id IN $ids)
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
        LIMIT $limit
        """
        rows = self._run(query, {"ids": candidate_ids, "limit": int(max_paths)})

        paths: List[Dict[str, Any]] = []
        for row in rows:
            paths.append(
                {
                    "hop_count": 4,
                    "path": [
                        {
                            "type": "person",
                            "id": row["person1_id"],
                            "name": row["person1_name"],
                        },
                        {
                            "type": "event",
                            "id": row["event1_id"],
                            "name": row["event1_name"],
                        },
                        {
                            "type": "person",
                            "id": row["person2_id"],
                            "name": row["person2_name"],
                        },
                        {
                            "type": "event",
                            "id": row["event2_id"],
                            "name": row["event2_name"],
                        },
                        {
                            "type": "person",
                            "id": row["person3_id"],
                            "name": row["person3_name"],
                        },
                        {
                            "type": "event",
                            "id": row["event3_id"],
                            "name": row["event3_name"],
                        },
                        {
                            "type": "person",
                            "id": row["person4_id"],
                            "name": row["person4_name"],
                        },
                    ],
                    "relationships": [
                        {
                            "type": row["rel1_type"],
                            "confidence": row.get("rel1_confidence", 0.5),
                            "evidence_text": row.get("rel1_evidence", ""),
                        },
                        {
                            "type": row["rel2_type"],
                            "confidence": row.get("rel2_confidence", 0.5),
                            "evidence_text": row.get("rel2_evidence", ""),
                        },
                        {
                            "type": row["rel3_type"],
                            "confidence": row.get("rel3_confidence", 0.5),
                            "evidence_text": row.get("rel3_evidence", ""),
                        },
                        {
                            "type": row["rel4_type"],
                            "confidence": row.get("rel4_confidence", 0.5),
                            "evidence_text": row.get("rel4_evidence", ""),
                        },
                        {
                            "type": row["rel5_type"],
                            "confidence": row.get("rel5_confidence", 0.5),
                            "evidence_text": row.get("rel5_evidence", ""),
                        },
                        {
                            "type": row["rel6_type"],
                            "confidence": row.get("rel6_confidence", 0.5),
                            "evidence_text": row.get("rel6_evidence", ""),
                        },
                    ],
                }
            )
        return paths


__all__ = [
    "GraphStore",
    "GraphNode",
    "GraphRelationship",
    "GraphPath",
]


