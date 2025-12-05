"""
Test script for retrieval module - compares retrieved paths vs ground truth paths.
This is faster than testing with LLM since we skip inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set
from chatbot.graph_store import GraphStore
from chatbot.graph_rag import retrieve_paths_for_question, build_evidence_from_path_data
from chatbot.graph_rag import build_evidence_from_example

DATA_FILE = Path(__file__).resolve().parents[0] / "data" / "multihop_questions.jsonl"


def load_example_by_id(q_id: str) -> Dict[str, Any] | None:
    """Load a question example by ID."""
    if not DATA_FILE.exists():
        return None
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            if example.get("id") == q_id:
                return example
    return None


def path_to_node_ids(path: List[Dict[str, Any]]) -> Set[int]:
    """Extract node IDs from a path."""
    return {node.get("id") for node in path if node.get("id")}


def calculate_path_overlap(retrieved_path: Dict[str, Any], ground_truth_path: Dict[str, Any]) -> float:
    """Calculate overlap between retrieved path and ground truth path.
    
    Returns a score from 0.0 to 1.0:
    - 1.0 if all ground truth nodes are in retrieved path
    - Partial score based on how many GT nodes are found
    """
    retrieved_ids = path_to_node_ids(retrieved_path.get("path", []))
    ground_truth_ids = path_to_node_ids(ground_truth_path.get("path", []))
    
    if not ground_truth_ids:
        return 0.0
    
    intersection = retrieved_ids & ground_truth_ids
    
    # Score based on how many GT nodes we found (precision on GT)
    # This is more important than Jaccard since we want to find all GT nodes
    return len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0.0


def test_retrieval_for_question(q_id: str, store: GraphStore) -> Dict[str, Any]:
    """Test retrieval for a single question and return metrics."""
    example = load_example_by_id(q_id)
    if not example:
        return {"error": f"Question {q_id} not found"}
    
    question_text = example.get("question", "").strip()
    hop_count = int(example.get("hop_count", 2))
    
    # Get ground truth path
    ground_truth_evidence = build_evidence_from_example(example)
    ground_truth_path = {
        "path": ground_truth_evidence.path,
        "relationships": ground_truth_evidence.relationships,
    }
    ground_truth_ids = path_to_node_ids(ground_truth_path.get("path", []))
    
    # Retrieve paths - get more paths for longer hops
    max_paths_to_retrieve = max(5, hop_count * 3)  # More paths for longer hops
    retrieved_paths = retrieve_paths_for_question(
        question_text=question_text,
        hop_hint=hop_count,
        store=store,
        max_paths=max_paths_to_retrieve,
    )
    
    # Calculate metrics
    best_overlap = 0.0
    best_path_idx = -1
    exact_match = False
    
    for idx, retrieved_path in enumerate(retrieved_paths):
        overlap = calculate_path_overlap(retrieved_path, ground_truth_path)
        if overlap > best_overlap:
            best_overlap = overlap
            best_path_idx = idx
        
        # Check for exact match
        retrieved_ids = path_to_node_ids(retrieved_path.get("path", []))
        if retrieved_ids == ground_truth_ids:
            exact_match = True
    
    # Check if any retrieved path contains all ground truth nodes
    contains_all = False
    for retrieved_path in retrieved_paths:
        retrieved_ids = path_to_node_ids(retrieved_path.get("path", []))
        if ground_truth_ids.issubset(retrieved_ids):
            contains_all = True
            break
    
    return {
        "q_id": q_id,
        "question": question_text[:60] + "..." if len(question_text) > 60 else question_text,
        "ground_truth_nodes": sorted(ground_truth_ids),
        "num_retrieved": len(retrieved_paths),
        "best_overlap": best_overlap,
        "exact_match": exact_match,
        "contains_all": contains_all,
        "best_path": retrieved_paths[best_path_idx] if best_path_idx >= 0 else None,
    }


def test_multiple_questions(q_ids: List[str] | None = None, max_questions: int = 200) -> None:
    """Test retrieval for multiple questions."""
    store = GraphStore()
    try:
        # Load question IDs
        if q_ids is None:
            q_ids = []
            if DATA_FILE.exists():
                with DATA_FILE.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        example = json.loads(line)
                        q_ids.append(example.get("id"))
                        if len(q_ids) >= max_questions:
                            break
        
        print(f"Testing retrieval for {len(q_ids)} questions...\n")
        
        results = []
        exact_matches = 0
        contains_all_count = 0
        total_overlap = 0.0
        
        from tqdm import tqdm
        
        for q_id in tqdm(q_ids, desc="Testing retrieval"):
            result = test_retrieval_for_question(q_id, store)
            if "error" in result:
                print(f"❌ {q_id}: {result['error']}")
                continue
            
            results.append(result)
            if result["exact_match"]:
                exact_matches += 1
            if result["contains_all"]:
                contains_all_count += 1
            total_overlap += result["best_overlap"]
            
            # Print result only for failures or every 10th question
            if not result["exact_match"] or len(results) % 10 == 0:
                status = "✅" if result["exact_match"] else ("✓" if result["contains_all"] else "⚠️")
                print(f"{status} {q_id}: overlap={result['best_overlap']:.2f}, "
                      f"retrieved={result['num_retrieved']}, "
                      f"GT nodes={len(result['ground_truth_nodes'])}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total questions: {len(results)}")
        print(f"  Exact matches: {exact_matches} ({exact_matches/len(results)*100:.1f}%)")
        print(f"  Contains all GT nodes: {contains_all_count} ({contains_all_count/len(results)*100:.1f}%)")
        print(f"  Average best overlap: {total_overlap/len(results):.2f}")
        print(f"{'='*60}")
        
        # Show worst cases
        print(f"\nWorst 5 retrievals (lowest overlap):")
        worst = sorted(results, key=lambda x: x["best_overlap"])[:5]
        for r in worst:
            print(f"  {r['q_id']}: overlap={r['best_overlap']:.2f}, question='{r['question']}'")
        
    finally:
        store.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific question IDs
        q_ids = sys.argv[1:]
        test_multiple_questions(q_ids)
    else:
        # Test all 2000 questions
        print("Testing all questions in dataset...")
        test_multiple_questions(max_questions=2000)

