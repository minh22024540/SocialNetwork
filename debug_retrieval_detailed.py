"""
Detailed debug script for retrieval issues.
"""

from chatbot.graph_store import GraphStore
from chatbot.graph_rag import retrieve_paths_for_question
from chatbot.alias_linker import link_entities_in_text
from chatbot.ner_extractor import extract_person_names, extract_event_names
import json

question = "Điểm chung nào sau đây giữa quân nhân Nguyễn Văn Tư và ông Trịnh Tố Tâm là đúng?"

print("="*60)
print("DEBUGGING RETRIEVAL FOR q_000012")
print("="*60)
print(f"\nQuestion: {question}\n")

store = GraphStore()

try:
    # Step 1: Entity linking
    print("Step 1: Entity Linking")
    print("-" * 60)
    ids_from_alias = link_entities_in_text(question, loose_match=True)
    print(f"Alias-linked IDs: {ids_from_alias[:10]}... (showing first 10)")
    
    # NER extraction
    ner_person_names = extract_person_names(question)
    ner_event_names = extract_event_names(question)
    print(f"NER person names: {ner_person_names}")
    print(f"NER event names: {ner_event_names}")
    
    # Check which IDs are persons
    person_ids_from_alias = []
    for pid in ids_from_alias:
        result = store._run(
            "MATCH (n) WHERE n.id = $id RETURN labels(n) AS labels, n.title AS title LIMIT 1",
            {"id": int(pid)}
        )
        if result:
            labels = result[0].get("labels", [])
            title = result[0].get("title", "")
            if "Event" not in labels:
                person_ids_from_alias.append(pid)
                if pid in [19573963, 19727592]:
                    print(f"  ✓ Found target person: {pid} ({title})")
    
    print(f"\nPerson IDs from alias: {person_ids_from_alias}")
    print(f"Target IDs should be: [19573963, 19727592]")
    
    # Step 2: Check if path exists in Neo4j
    print("\n" + "="*60)
    print("Step 2: Check if path exists in Neo4j")
    print("-" * 60)
    result = store._run(
        """
        MATCH path = (p1)-[r1:HAS_RELATIONSHIP]->(e:Event)<-[r2:HAS_RELATIONSHIP]-(p2)
        WHERE p1.id IN [19573963, 19727592] AND p2.id IN [19573963, 19727592] AND p1 <> p2
        RETURN p1.id AS p1_id, p1.title AS p1_name, 
               e.id AS e_id, e.title AS e_name,
               p2.id AS p2_id, p2.title AS p2_name
        LIMIT 5
        """,
        {}
    )
    print(f"Paths found in Neo4j: {len(result)}")
    for r in result:
        print(f"  {r['p1_name']} (id={r['p1_id']}) -> {r['e_name']} (id={r['e_id']}) <- {r['p2_name']} (id={r['p2_id']})")
    
    # Step 3: Test the connecting query directly
    print("\n" + "="*60)
    print("Step 3: Test connecting query with person_ids")
    print("-" * 60)
    test_person_ids = [19573963, 19727592]
    query_result = store._run(
        """
        MATCH path = (p1)-[r1:HAS_RELATIONSHIP]->(e:Event)<-[r2:HAS_RELATIONSHIP]-(p2)
        WHERE p1.id IN $person_ids AND p2.id IN $person_ids AND p1 <> p2
        RETURN p1.id AS person1_id, p1.title AS person1_name,
               e.id AS event_id, e.title AS event_name,
               p2.id AS person2_id, p2.title AS person2_name
        LIMIT 10
        """,
        {"person_ids": test_person_ids}
    )
    print(f"Connecting query results: {len(query_result)}")
    for r in query_result:
        print(f"  {r['person1_name']} -> {r['event_name']} <- {r['person2_name']}")
    
    # Step 4: Test the actual retrieval function
    print("\n" + "="*60)
    print("Step 4: Test retrieve_paths_for_question")
    print("-" * 60)
    paths = retrieve_paths_for_question(
        question_text=question,
        hop_hint=2,
        store=store,
        max_paths=10,
    )
    print(f"Retrieved {len(paths)} paths")
    for i, path in enumerate(paths[:5], 1):
        path_ids = [node.get("id") for node in path.get("path", [])]
        path_names = [node.get("name") for node in path.get("path", [])]
        has_target = any(pid in [19573963, 19727592] for pid in path_ids)
        print(f"  Path {i}: {path_names}")
        print(f"    IDs: {path_ids}")
        print(f"    Contains target persons: {has_target}")
    
    # Step 5: Check what person_ids were passed to sample_paths_for_candidates_2hop
    print("\n" + "="*60)
    print("Step 5: Check internal state")
    print("-" * 60)
    # We need to see what's happening inside retrieve_paths_for_question
    # Let's check the code flow
    
finally:
    store.close()

