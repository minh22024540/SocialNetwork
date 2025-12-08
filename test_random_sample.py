"""
Test retrieval on a random sample of questions.
"""

import json
import random
from pathlib import Path
from chatbot.graph_store import GraphStore
from chatbot.graph_rag import retrieve_paths_for_question, build_evidence_from_example
from test_retrieval import path_to_node_ids, calculate_path_overlap

DATA_FILE = Path(__file__).resolve().parents[0] / "data" / "multihop_questions.jsonl"

def test_random_questions(n=20, seed=42):
    """Test retrieval on n random questions."""
    random.seed(seed)
    
    # Load all question IDs
    all_q_ids = []
    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                all_q_ids.append(example.get("id"))
    
    # Sample randomly
    sample_q_ids = random.sample(all_q_ids, min(n, len(all_q_ids)))
    print(f"Testing {len(sample_q_ids)} randomly selected questions...\n")
    
    store = GraphStore()
    try:
        results = []
        exact_matches = 0
        contains_all_count = 0
        total_overlap = 0.0
        
        from tqdm import tqdm
        
        for q_id in tqdm(sample_q_ids, desc="Testing"):
            # Load question
            example = None
            with DATA_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    if ex.get("id") == q_id:
                        example = ex
                        break
            
            if not example:
                continue
            
            question_text = example.get("question", "").strip()
            hop_count = int(example.get("hop_count", 2))
            
            # Get ground truth
            ground_truth_evidence = build_evidence_from_example(example)
            ground_truth_path = {
                "path": ground_truth_evidence.path,
                "relationships": ground_truth_evidence.relationships,
            }
            ground_truth_ids = path_to_node_ids(ground_truth_path.get("path", []))
            
            # Retrieve paths
            max_paths_to_retrieve = max(5, hop_count * 3)
            retrieved_paths = retrieve_paths_for_question(
                question_text=question_text,
                hop_hint=hop_count,
                store=store,
                max_paths=max_paths_to_retrieve,
            )
            
            # Calculate metrics
            best_overlap = 0.0
            exact_match = False
            contains_all = False
            
            for retrieved_path in retrieved_paths:
                overlap = calculate_path_overlap(retrieved_path, ground_truth_path)
                if overlap > best_overlap:
                    best_overlap = overlap
                
                retrieved_ids = path_to_node_ids(retrieved_path.get("path", []))
                if retrieved_ids == ground_truth_ids:
                    exact_match = True
                if ground_truth_ids.issubset(retrieved_ids):
                    contains_all = True
            
            results.append({
                "q_id": q_id,
                "question": question_text[:60] + "..." if len(question_text) > 60 else question_text,
                "best_overlap": best_overlap,
                "exact_match": exact_match,
                "contains_all": contains_all,
            })
            
            if exact_match:
                exact_matches += 1
            if contains_all:
                contains_all_count += 1
            total_overlap += best_overlap
            
            # Print failures immediately
            if not exact_match:
                status = "✓" if contains_all else "⚠️"
                print(f"\n{status} {q_id}: overlap={best_overlap:.2f}, question='{question_text[:60]}...'")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total questions: {len(results)}")
        print(f"  Exact matches: {exact_matches} ({exact_matches/len(results)*100:.1f}%)")
        print(f"  Contains all GT nodes: {contains_all_count} ({contains_all_count/len(results)*100:.1f}%)")
        print(f"  Average best overlap: {total_overlap/len(results):.2f}")
        print(f"{'='*60}")
        
        # Show worst cases
        worst = sorted(results, key=lambda x: x["best_overlap"])[:5]
        if worst and worst[0]["best_overlap"] < 1.0:
            print(f"\nWorst 5 retrievals (lowest overlap):")
            for r in worst:
                print(f"  {r['q_id']}: overlap={r['best_overlap']:.2f}, question='{r['question']}'")
        
    finally:
        store.close()

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    test_random_questions(n=n)

