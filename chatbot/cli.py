"""
Simple CLI entrypoint to interact with the GraphRAG + Qwen chatbot.

Supports two modes:
1. --id: Load a question from the benchmark dataset
2. --question: Accept a free-form question and retrieve paths from Neo4j
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so we can import chatbot modules
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from chatbot.graph_store import GraphStore
from chatbot.graph_rag import (
    build_chat_prompt_for_question,
    build_chat_prompt_for_free_question,
    build_evidence_from_path_data,
    retrieve_paths_for_question,
)
from chatbot.ollama_client import OllamaClient, LLMMessage


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "multihop_questions.jsonl"


def load_example_by_id(q_id: str) -> Optional[dict]:
    """Load a question example from the benchmark dataset by ID."""
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        return None
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id") == q_id:
                return obj
    return None


def chat_with_example(q_id: str) -> None:
    """Chat using a question from the benchmark dataset."""
    example = load_example_by_id(q_id)
    if example is None:
        print(f"Question with id={q_id} not found in {DATA_FILE}")
        return

    store = GraphStore()
    client = OllamaClient()
    try:
        question_text = example.get("question", "").strip()
        q_type = example.get("type", "true_false")
        hop_count = int(example.get("hop_count", 2))
        
        # Retrieve paths from Neo4j (like evaluate_chatbot does)
        print(f"Retrieving paths from knowledge graph for question: {question_text[:50]}...")
        path_dicts = retrieve_paths_for_question(
            question_text=question_text,
            hop_hint=hop_count,
            store=store,
            max_paths=5,
        )
        
        print(f"Found {len(path_dicts)} path(s)\n")
        
        # Build evidence from retrieved paths
        evidences = [build_evidence_from_path_data(p) for p in path_dicts]
        
        # Build prompt using retrieved paths
        prompt_parts = build_chat_prompt_for_free_question(
            question_text=question_text,
            q_type=q_type,
            hop_count=hop_count,
            evidences=evidences,
        )
        system_prompt = prompt_parts["system_prompt"]
        user_content = prompt_parts["user_content"]

        print("=== Graph context + question sent to model ===")
        print(user_content)
        print("==============================================\n")

        resp = client.generate(
            system_prompt=system_prompt,
            messages=[LLMMessage(role="user", content=user_content)],
            temperature=0.1,
            max_tokens=256,
        )

        print("=== Model answer ===")
        print(resp.text)
        print("====================")
    finally:
        store.close()
        client.close()


def chat_with_free_question(question_text: str, hop_hint: int = 2) -> None:
    """Chat using a free-form question with GraphRAG retrieval from Neo4j."""
    store = GraphStore()
    client = OllamaClient()
    
    try:
        print(f"Question: {question_text}\n")
        print("Retrieving relevant paths from knowledge graph...")
        
        # Retrieve paths from Neo4j
        path_dicts = retrieve_paths_for_question(
            question_text=question_text,
            hop_hint=hop_hint,
            store=store,
            max_paths=5,
        )
        
        if not path_dicts:
            print("⚠️  No relevant paths found in the knowledge graph for this question.")
            print("This might mean:")
            print("  - The entities mentioned are not in the graph")
            print("  - There are no multi-hop paths connecting them")
            return
        
        print(f"Found {len(path_dicts)} relevant path(s)\n")
        
        # Build evidence objects
        evidences = [build_evidence_from_path_data(p) for p in path_dicts]
        
        # Build prompt (assume general question type, not true_false/yes_no/mcq)
        prompt_parts = build_chat_prompt_for_free_question(
            question_text=question_text,
            q_type="general",  # General question, not constrained to specific format
            hop_count=hop_hint,
            evidences=evidences,
        )
        system_prompt = prompt_parts["system_prompt"]
        user_content = prompt_parts["user_content"]
        
        print("=== Graph context retrieved from knowledge graph ===")
        print(user_content)
        print("====================================================\n")
        
        print("Generating answer...\n")
        resp = client.generate(
            system_prompt=system_prompt,
            messages=[LLMMessage(role="user", content=user_content)],
            temperature=0.1,
            max_tokens=512,
        )
        
        print("=== Model answer ===")
        print(resp.text)
        print("====================")
    finally:
        store.close()
        client.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Interact with the GraphRAG + Qwen chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a question from the benchmark dataset
  python -m SocialNetwork.chatbot.cli --id q_000123
  
  # Ask a free-form question
  python -m SocialNetwork.chatbot.cli --question "Hồ Chí Minh và Võ Nguyên Giáp có liên quan gì?"
  
  # Ask with a specific hop hint (2, 3, or 4)
  python -m SocialNetwork.chatbot.cli --question "Ai đã tham gia Chiến tranh Việt Nam?" --hop 3
        """,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--id",
        dest="q_id",
        help="Question ID in multihop_questions.jsonl (e.g. q_000123)",
    )
    group.add_argument(
        "--question",
        dest="question_text",
        help="Free-form question to ask the chatbot",
    )
    
    parser.add_argument(
        "--hop",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Hint for multi-hop reasoning depth (default: 2). Only used with --question",
    )
    
    args = parser.parse_args()
    
    if args.q_id:
        chat_with_example(args.q_id)
    elif args.question_text:
        chat_with_free_question(args.question_text, hop_hint=args.hop)


if __name__ == "__main__":
    main()


