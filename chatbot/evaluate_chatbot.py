"""
Offline evaluation of the GraphRAG + Qwen2.5-0.5B chatbot on the 2000-question
multi-hop benchmark stored in `multihop_questions.jsonl`.

Usage (after starting Ollama with the Qwen model, e.g. `ollama run qwen2.5:0.5b`):

    cd /home/ubuntu/Videos && source .venv/bin/activate
    python code/SocialNetwork/chatbot/evaluate_chatbot.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Add parent directory to path so we can import chatbot modules
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from chatbot.graph_store import GraphStore
from chatbot.graph_rag import (
    build_chat_prompt_for_free_question,
    build_evidence_from_path_data,
    retrieve_paths_for_question,
)
from chatbot.ollama_client import OllamaClient, LLMMessage


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "multihop_questions.jsonl"


@dataclass
class EvalStats:
    total: int = 0
    correct: int = 0
    by_hop: Dict[int, Dict[str, int]] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def update(self, hop_count: int, q_type: str, is_correct: bool) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1

        hop_bucket = self.by_hop.setdefault(hop_count, {"total": 0, "correct": 0})
        hop_bucket["total"] += 1
        if is_correct:
            hop_bucket["correct"] += 1

        type_bucket = self.by_type.setdefault(q_type, {"total": 0, "correct": 0})
        type_bucket["total"] += 1
        if is_correct:
            type_bucket["correct"] += 1


def _normalize_answer(raw: str, q_type: str) -> str:
    """Map free-form model output to canonical label."""
    text = raw.strip().lower()

    if q_type == "true_false":
        if text.startswith("đúng") or text.startswith("true"):
            return "True"
        if text.startswith("sai") or text.startswith("false"):
            return "False"
    elif q_type == "yes_no":
        if text.startswith("có") or text.startswith("yes"):
            return "Yes"
        if text.startswith("không") or text.startswith("no"):
            return "No"
    elif q_type == "multiple_choice":
        for opt in ["a", "b", "c", "d"]:
            if text.startswith(opt):
                return opt.upper()

    # Fallback: return the first token uppercased (for MCQ) or capitalized
    if q_type == "multiple_choice" and text:
        return text[0].upper()
    return raw.strip()


def evaluate(
    client: OllamaClient,
    max_examples: int | None = None,
) -> None:
    stats = EvalStats()

    store = GraphStore()

    try:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

        # Count total lines for progress bar
        print("Counting questions...")
        with DATA_FILE.open("r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())
        
        if max_examples is not None:
            total_lines = min(total_lines, max_examples)
        
        print(f"Evaluating {total_lines} questions...\n")

        with DATA_FILE.open("r", encoding="utf-8") as f:
            pbar = tqdm(enumerate(f, 1), total=total_lines, desc="Evaluating", unit="question")
            for idx, line in pbar:
                if max_examples is not None and stats.total >= max_examples:
                    break
                if not line.strip():
                    continue

                example = json.loads(line)
                q_type = example.get("type", "true_false")
                hop_count = int(example.get("hop_count", 2))
                correct_answer = str(example.get("correct_answer", "")).strip()
                question_text = str(example.get("question", "")).strip()

                # GraphRAG: retrieve paths from Neo4j purely from the question text.
                path_dicts = retrieve_paths_for_question(
                    question_text=question_text,
                    hop_hint=hop_count,
                    store=store,
                    max_paths=5,
                )
                evidences = [build_evidence_from_path_data(p) for p in path_dicts]

                prompt_parts = build_chat_prompt_for_free_question(
                    question_text=question_text,
                    q_type=q_type,
                    hop_count=hop_count,
                    evidences=evidences,
                )
                system_prompt = prompt_parts["system_prompt"]
                user_content = prompt_parts["user_content"]

                resp = client.generate(
                    system_prompt=system_prompt,
                    messages=[LLMMessage(role="user", content=user_content)],
                    temperature=0.1,
                    max_tokens=256,
                )

                predicted = _normalize_answer(resp.text, q_type)
                is_correct = predicted == correct_answer
                stats.update(hop_count, q_type, is_correct)

                # Update progress bar with current accuracy
                current_acc = stats.correct / max(stats.total, 1)
                pbar.set_postfix({
                    "accuracy": f"{current_acc:.3f}",
                    "correct": f"{stats.correct}/{stats.total}"
                })
    finally:
        store.close()

    print("\n=== Evaluation finished ===")
    if stats.total:
        overall_acc = stats.correct / stats.total
    else:
        overall_acc = 0.0
    print(f"Overall accuracy: {stats.correct}/{stats.total} = {overall_acc:.3f}")

    print("\nAccuracy by hop count:")
    for hop, bucket in sorted(stats.by_hop.items()):
        acc = bucket["correct"] / max(bucket["total"], 1)
        print(f"  {hop}-hop: {bucket['correct']}/{bucket['total']} = {acc:.3f}")

    print("\nAccuracy by question type:")
    for q_type, bucket in sorted(stats.by_type.items()):
        acc = bucket["correct"] / max(bucket["total"], 1)
        print(f"  {q_type}: {bucket['correct']}/{bucket['total']} = {acc:.3f}")


def main() -> None:
    client = OllamaClient()
    try:
        evaluate(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()


