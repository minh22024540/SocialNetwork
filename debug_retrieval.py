"""Debug script to investigate retrieval issues for specific questions."""

import json
from pathlib import Path
from chatbot.graph_store import GraphStore
from chatbot.graph_rag import retrieve_paths_for_question, build_evidence_from_example
from chatbot.alias_linker import link_entities_in_text

DATA_FILE = Path(__file__).resolve().parents[0] / "data" / "multihop_questions.jsonl"


def load_example_by_id(q_id: str):
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            if example.get("id") == q_id:
                return example
    return None


def debug_question(q_id: str):
    example = load_example_by_id(q_id)
    if not example:
        print(f"Question {q_id} not found")
        return
    
    question_text = example.get("question", "")
    hop_count = int(example.get("hop_count", 2))
    
    print(f"Question: {question_text}")
    print(f"Hop count: {hop_count}\n")
    
    # Ground truth
    gt_evidence = build_evidence_from_example(example)
    print("Ground Truth Path:")
    for i, node in enumerate(gt_evidence.path):
        print(f"  Node {i+1}: {node.get('name')} (id={node.get('id')})")
    gt_ids = {node.get('id') for node in gt_evidence.path}
    print(f"GT Node IDs: {sorted(gt_ids)}\n")
    
    # Entity linking
    linked_ids = link_entities_in_text(question_text, loose_match=True)
    print(f"Linked Entity IDs: {linked_ids}\n")
    
    # Retrieval
    store = GraphStore()
    try:
        retrieved_paths = retrieve_paths_for_question(
            question_text=question_text,
            hop_hint=hop_count,
            store=store,
            max_paths=10,
        )
        
        print(f"Retrieved {len(retrieved_paths)} paths:\n")
        for idx, path in enumerate(retrieved_paths[:5], 1):
            path_ids = {node.get('id') for node in path.get('path', [])}
            overlap = len(path_ids & gt_ids) / len(gt_ids) if gt_ids else 0
            print(f"Path {idx} (overlap={overlap:.2f}):")
            for i, node in enumerate(path.get('path', [])):
                marker = "✓" if node.get('id') in gt_ids else " "
                print(f"  {marker} Node {i+1}: {node.get('name')} (id={node.get('id')})")
            print()
    finally:
        store.close()


if __name__ == "__main__":
    import sys
    q_id = sys.argv[1] if len(sys.argv) > 1 else "q_001889"
    debug_question(q_id)

