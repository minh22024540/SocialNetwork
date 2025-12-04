"""GraphRAG utilities for the social network knowledge graph.

Formats paths and relationships into LLM-friendly text, builds prompts for
GraphRAG-style question answering, and retrieves relevant multi-hop paths
(2–4 hops) from Neo4j via GraphStore for free-form questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from SocialNetwork.chatbot.graph_store import GraphStore
from SocialNetwork.chatbot.alias_linker import link_entities_in_text


@dataclass
class GraphEvidence:
    hop_count: int
    path: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


def build_evidence_from_path_data(path_data: Dict[str, Any]) -> GraphEvidence:
    """Build GraphEvidence from a path dict (same schema as sampling scripts)."""
    hop_count = int(path_data.get("hop_count", 2))
    path = path_data.get("path") or []
    relationships = path_data.get("relationships") or []
    return GraphEvidence(hop_count=hop_count, path=path, relationships=relationships)


def build_evidence_from_example(example: Dict[str, Any]) -> GraphEvidence:
    """Extract graph evidence (path + relationships) from a question example."""
    hop_count = int(example.get("hop_count", 2))
    path = example.get("source_path") or example.get("path") or []
    relationships = example.get("relationships") or []
    return GraphEvidence(hop_count=hop_count, path=path, relationships=relationships)


def format_graph_evidence_for_prompt(evidence: GraphEvidence) -> str:
    """Format a single path + relationships into a compact, LLM-friendly string.
    
    Returns Vietnamese-formatted string for use in Vietnamese prompts.
    """
    nodes = evidence.path
    rels = evidence.relationships

    # Vietnamese message: "No sufficient information about path in graph"
    if not nodes or not rels:
        return "Không có đủ thông tin về đường đi trong đồ thị."

    lines: List[str] = []
    # Vietnamese: "{hop_count}-hop path in knowledge graph"
    lines.append(f"Đường đi {evidence.hop_count}-hop trong đồ thị tri thức:")

    # Node descriptions
    for idx, node in enumerate(nodes):
        ntype = node.get("type", "")
        nid = node.get("id", "")
        name = node.get("name", "")
        lines.append(f"- Node {idx+1}: [{ntype}] {name} (id={nid})")

    # Relationship descriptions (aligned between consecutive nodes)
    lines.append("")
    # Vietnamese: "Relationships between consecutive nodes"
    lines.append("Các quan hệ giữa các node liên tiếp:")
    for i, rel in enumerate(rels):
        if i >= len(nodes) - 1:
            break
        src = nodes[i]
        dst = nodes[i + 1]
        rtype = rel.get("type", "")
        conf = rel.get("confidence", 0.5)
        ev = (rel.get("evidence_text") or "").strip()
        src_name = src.get("name", "")
        dst_name = dst.get("name", "")
        # Vietnamese format: "source --[type, confidence ~X.XX]--> destination"
        # "độ tin cậy" = confidence, "Bằng chứng" = evidence
        base = f"{src_name} --[{rtype}, độ tin cậy ~{conf:.2f}]--> {dst_name}"
        if ev:
            lines.append(f"- {base}. Bằng chứng: \"{ev}\"")
        else:
            lines.append(f"- {base}.")

    return "\n".join(lines)


def build_chat_prompt_for_question(example: Dict[str, Any]) -> Dict[str, str]:
    """Construct system_prompt and user_content for Qwen given one question row."""
    evidence = build_evidence_from_example(example)
    graph_context = format_graph_evidence_for_prompt(evidence)
    question_text = example.get("question", "").strip()

    system_prompt = (
        "Bạn là một trợ lý trả lời câu hỏi lịch sử dựa trên ĐỒ THỊ TRI THỨC về con người "
        "và sự kiện.\n\n"
        "QUAN TRỌNG:\n"
        "- Chỉ được sử dụng thông tin có trong phần mô tả đồ thị bên dưới (các node, "
        "quan hệ, evidence_text).\n"
        "- Phải suy luận multi-hop: kết hợp lần lượt các quan hệ trong đường đi đã cho.\n"
        "- Không được thêm kiến thức bên ngoài đồ thị, không bịa thêm chi tiết.\n"
        "- Trả lời ngắn gọn và trực tiếp vào câu hỏi, sau đó giải thích suy luận dựa "
        "trên các quan hệ trong đồ thị.\n"
    )

    user_content = (
        f"{graph_context}\n\n"
        "Dựa vào đồ thị tri thức ở trên, hãy trả lời câu hỏi sau bằng tiếng Việt:\n"
        f"Câu hỏi: {question_text}\n\n"
        "Yêu cầu:\n"
        "- Nếu câu hỏi là Đúng/Sai (true_false), hãy trả lời bằng một từ: 'Đúng' hoặc 'Sai', "
        "sau đó giải thích ngắn gọn.\n"
        "- Nếu câu hỏi là Có/Không (yes_no), hãy trả lời bằng một từ: 'Có' hoặc 'Không', "
        "sau đó giải thích ngắn gọn.\n"
        "- Nếu câu hỏi là trắc nghiệm, hãy trả lời bằng một chữ cái duy nhất (A, B, C hoặc D), "
        "sau đó giải thích ngắn gọn tại sao lựa chọn đó đúng.\n"
    )

    return {"system_prompt": system_prompt, "user_content": user_content}


def format_multiple_evidences_for_prompt(evidences: List[GraphEvidence]) -> str:
    """Format multiple paths as a multi-block graph context string.
    
    Returns Vietnamese-formatted string for use in Vietnamese prompts.
    """
    if not evidences:
        # Vietnamese: "No relevant paths found in knowledge graph for this question.
        # If information is missing, clearly state that the graph doesn't provide enough data."
        return (
            "Không tìm được đường đi liên quan trong đồ thị tri thức cho câu hỏi này. "
            "Nếu thiếu thông tin, hãy nói rõ rằng đồ thị không cung cấp đủ dữ liệu."
        )

    parts: List[str] = []
    for idx, ev in enumerate(evidences):
        parts.append(f"=== Đường đi {idx + 1} ({ev.hop_count}-hop) ===")
        parts.append(format_graph_evidence_for_prompt(ev))
        parts.append("")
    return "\n".join(parts).rstrip()


def build_chat_prompt_for_free_question(
    question_text: str,
    q_type: str,
    hop_count: int,
    evidences: List[GraphEvidence],
) -> Dict[str, str]:
    """Build system and user prompts for a free-form question using retrieved paths.

    Args:
        question_text: Free-form question string from user.
        q_type: Question type ("true_false", "yes_no", "multiple_choice", or "general").
        hop_count: Expected number of hops in reasoning path.
        evidences: List of GraphEvidence objects from path retrieval.

    Returns:
        Dictionary with "system_prompt" and "user_content" keys for LLM chat API.
    """
    graph_context = format_multiple_evidences_for_prompt(evidences)

    # Vietnamese system prompt: instructs LLM to answer history questions based on knowledge graph
    # "Bạn là một trợ lý..." = "You are an assistant..."
    # "QUAN TRỌNG" = "IMPORTANT"
    system_prompt = (
        "Bạn là một trợ lý trả lời câu hỏi lịch sử dựa trên ĐỒ THỊ TRI THỨC về con người "
        "và sự kiện.\n\n"
        "QUAN TRỌNG:\n"
        "- Chỉ được sử dụng thông tin có trong phần mô tả đồ thị bên dưới (các node, "
        "quan hệ, evidence_text).\n"
        "- Phải suy luận multi-hop: kết hợp lần lượt các quan hệ trong các đường đi đã cho.\n"
        "- Không được thêm kiến thức bên ngoài đồ thị, không bịa thêm chi tiết.\n"
        "- Trả lời ngắn gọn và trực tiếp vào câu hỏi, sau đó giải thích suy luận dựa "
        "trên các quan hệ trong đồ thị.\n"
    )

    # Build answer format instructions in Vietnamese based on question type
    format_instructions = ""
    if q_type == "true_false":
        # "Answer with one word: 'True' or 'False', then briefly explain"
        format_instructions = "Hãy trả lời bằng một từ: 'Đúng' hoặc 'Sai', sau đó giải thích ngắn gọn.\n"
    elif q_type == "yes_no":
        # "Answer with one word: 'Yes' or 'No', then briefly explain"
        format_instructions = "Hãy trả lời bằng một từ: 'Có' hoặc 'Không', sau đó giải thích ngắn gọn.\n"
    elif q_type == "multiple_choice":
        # "Answer with a single letter (A, B, C or D), then briefly explain why that choice is correct"
        format_instructions = "Hãy trả lời bằng một chữ cái duy nhất (A, B, C hoặc D), sau đó giải thích ngắn gọn tại sao lựa chọn đó đúng.\n"
    else:
        # General question - just answer naturally
        # "Answer naturally and briefly, then explain reasoning based on relationships in graph"
        format_instructions = "Hãy trả lời một cách tự nhiên và ngắn gọn, sau đó giải thích suy luận dựa trên các quan hệ trong đồ thị.\n"

    # Vietnamese user content: "Based on the knowledge graph above, answer the following question in Vietnamese"
    user_content = (
        f"{graph_context}\n\n"
        "Dựa vào đồ thị tri thức ở trên, hãy trả lời câu hỏi sau bằng tiếng Việt:\n"
        f"Câu hỏi: {question_text}\n\n"
        f"Yêu cầu: {format_instructions}"
    )

    return {"system_prompt": system_prompt, "user_content": user_content}


def retrieve_paths_for_question(
    question_text: str,
    hop_hint: int,
    store: GraphStore,
    max_paths: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve candidate multi-hop paths from Neo4j for a free-form question."""
    hop_hint = int(hop_hint)
    if hop_hint < 2:
        hop_hint = 2
    if hop_hint > 4:
        hop_hint = 4

    # 1) Link entities using alias JSON + fuzzy title matching in Neo4j.
    ids_from_alias = link_entities_in_text(question_text)
    nodes_from_title = store.find_nodes_in_text(
        text=question_text,
        labels=["Person", "Event"],
        limit=20,
    )
    ids_from_title = [int(n.id) for n in nodes_from_title]

    candidate_ids = sorted({*ids_from_alias, *ids_from_title})

    if not candidate_ids:
        return []

    # 2) Query paths of the appropriate hop length that touch any candidate.
    if hop_hint == 2:
        return store.sample_paths_for_candidates_2hop(candidate_ids, max_paths)
    if hop_hint == 3:
        return store.sample_paths_for_candidates_3hop(candidate_ids, max_paths)
    return store.sample_paths_for_candidates_4hop(candidate_ids, max_paths)


__all__ = [
    "GraphEvidence",
    "build_evidence_from_path_data",
    "build_evidence_from_example",
    "format_graph_evidence_for_prompt",
    "build_chat_prompt_for_question",
    "format_multiple_evidences_for_prompt",
    "build_chat_prompt_for_free_question",
    "retrieve_paths_for_question",
]


