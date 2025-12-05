"""GraphRAG utilities for the social network knowledge graph.

This module provides functions to:
- Format graph paths and relationships into LLM-friendly text
- Build prompts for GraphRAG-style question answering
- Retrieve relevant multi-hop paths (2-4 hops) from Neo4j
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from chatbot.alias_linker import link_entities_in_text
from chatbot.graph_store import GraphStore


@dataclass
class GraphEvidence:
    """Represents a single path through the knowledge graph.

    Attributes:
        hop_count: Number of hops in the path.
        path: List of nodes in the path, each with id, name, type.
        relationships: List of relationships between consecutive nodes.
    """

    hop_count: int
    path: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


def build_evidence_from_path_data(path_data: Dict[str, Any]) -> GraphEvidence:
    """Build GraphEvidence from a path dictionary.

    Args:
        path_data: Dictionary with keys 'hop_count', 'path', 'relationships'.

    Returns:
        GraphEvidence object constructed from the path data.
    """
    hop_count = int(path_data.get("hop_count", 2))
    path = path_data.get("path") or []
    relationships = path_data.get("relationships") or []
    return GraphEvidence(hop_count=hop_count, path=path, relationships=relationships)


def build_evidence_from_example(example: Dict[str, Any]) -> GraphEvidence:
    """Extract graph evidence from a question example dictionary.

    Args:
        example: Dictionary containing question data with 'hop_count', 'source_path'
            or 'path', and 'relationships' keys.

    Returns:
        GraphEvidence object extracted from the example.
    """
    hop_count = int(example.get("hop_count", 2))
    path = example.get("source_path") or example.get("path") or []
    relationships = example.get("relationships") or []
    return GraphEvidence(hop_count=hop_count, path=path, relationships=relationships)


def format_graph_evidence_for_prompt(evidence: GraphEvidence) -> str:
    """Format a single path and relationships into LLM-friendly Vietnamese text.

    Args:
        evidence: GraphEvidence object containing path and relationships.

    Returns:
        Formatted Vietnamese string describing the path and relationships.
    """
    nodes = evidence.path
    rels = evidence.relationships

    if not nodes or not rels:
        return "Không có đủ thông tin về đường đi trong đồ thị."

    lines: List[str] = []
    lines.append(f"Đường đi {evidence.hop_count}-hop trong đồ thị tri thức:")

    for idx, node in enumerate(nodes):
        ntype = node.get("type", "")
        nid = node.get("id", "")
        name = node.get("name", "")
        lines.append(f"- Node {idx+1}: [{ntype}] {name} (id={nid})")

    lines.append("")
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
        base = f"{src_name} --[{rtype}, độ tin cậy ~{conf:.2f}]--> {dst_name}"
        if ev:
            lines.append(f"- {base}. Bằng chứng: \"{ev}\"")
        else:
            lines.append(f"- {base}.")

    return "\n".join(lines)


def build_chat_prompt_for_question(example: Dict[str, Any]) -> Dict[str, str]:
    """Construct system and user prompts for a question from example data.

    Args:
        example: Dictionary containing question data with 'question', 'type',
            'hop_count', 'source_path' or 'path', and 'relationships' keys.

    Returns:
        Dictionary with 'system_prompt' and 'user_content' keys for LLM chat API.
    """
    evidence = build_evidence_from_example(example)
    graph_context = format_graph_evidence_for_prompt(evidence)
    question_text = example.get("question", "").strip()
    q_type = example.get("type", "true_false")

    system_prompt = (
        "Bạn trả lời câu hỏi dựa trên ĐỒ THỊ TRI THỨC (knowledge graph). "
        "Nếu đồ thị viết 'Không có đủ thông tin', trả lời: 'Sai. Không có đủ thông tin trong đồ thị để xác nhận.' "
        "hoặc 'Không. Không có đủ thông tin trong đồ thị để xác nhận.' (tùy loại câu hỏi). "
        "Nếu có thông tin trong đồ thị, trả lời 'Đúng' hoặc 'Sai' rồi giải thích trong 1 câu dựa trên đồ thị. "
        "Bắt đầu bằng 'Đúng', 'Sai', 'Có', hoặc 'Không'. Dừng ngay sau câu giải thích. "
        "QUAN TRỌNG: Dùng từ 'Không' (No), KHÔNG dùng 'Não' (brain). Dùng từ 'đồ thị' (graph), KHÔNG dùng 'não'."
    )

    has_info = graph_context and "Không có đủ thông tin" not in graph_context

    if not has_info:
        answer_keyword = "Sai" if q_type == "true_false" else "Không" if q_type == "yes_no" else "Không có đủ thông tin"
        user_content = (
            f"Đồ thị: {graph_context}\n\n"
            f"Câu hỏi: {question_text}\n\n"
            f"Trả lời: Bắt đầu bằng từ '{answer_keyword}' (KHÔNG dùng 'Não'). "
            f"Sau đó giải thích: 'Không có đủ thông tin trong đồ thị để xác nhận.'\n"
        )
    else:
        format_instructions = ""
        if q_type == "true_false":
            format_instructions = "Hãy trả lời bằng một từ: 'Đúng' hoặc 'Sai', sau đó giải thích ngắn gọn.\n"
        elif q_type == "yes_no":
            format_instructions = "Hãy trả lời bằng một từ: 'Có' hoặc 'Không', sau đó giải thích ngắn gọn.\n"
        elif q_type == "multiple_choice":
            format_instructions = "Hãy trả lời bằng một chữ cái duy nhất (A, B, C hoặc D), sau đó giải thích ngắn gọn tại sao lựa chọn đó đúng.\n"

        user_content = (
            f"{graph_context}\n\n"
            f"Câu hỏi: {question_text}\n\n"
            f"Trả lời:\n"
        )

    return {"system_prompt": system_prompt, "user_content": user_content}


def format_multiple_evidences_for_prompt(evidences: List[GraphEvidence]) -> str:
    """Format multiple paths as a multi-block graph context string.

    Args:
        evidences: List of GraphEvidence objects to format.

    Returns:
        Formatted Vietnamese string containing all paths.
    """
    if not evidences:
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

    system_prompt = (
        "Trả lời câu hỏi dựa trên đồ thị. "
        "Nếu đồ thị viết 'Không có đủ thông tin', trả lời: 'Sai. Không có đủ thông tin trong đồ thị để xác nhận.' "
        "Nếu có thông tin, trả lời 'Đúng' hoặc 'Sai' rồi giải thích ngắn gọn. "
        "Bắt đầu bằng 'Đúng' hoặc 'Sai'."
    )

    format_instructions = ""
    if q_type == "true_false":
        format_instructions = "Hãy trả lời bằng một từ: 'Đúng' hoặc 'Sai', sau đó giải thích ngắn gọn.\n"
    elif q_type == "yes_no":
        format_instructions = "Hãy trả lời bằng một từ: 'Có' hoặc 'Không', sau đó giải thích ngắn gọn.\n"
    elif q_type == "multiple_choice":
        format_instructions = "Hãy trả lời bằng một chữ cái duy nhất (A, B, C hoặc D), sau đó giải thích ngắn gọn tại sao lựa chọn đó đúng.\n"
    else:
        format_instructions = "Hãy trả lời một cách tự nhiên và ngắn gọn, sau đó giải thích suy luận dựa trên các quan hệ trong đồ thị.\n"

    has_info = (
        graph_context
        and "Không có đủ thông tin" not in graph_context
        and "Không tìm được đường đi" not in graph_context
    )

    if not has_info:
        answer_keyword = "Sai" if q_type == "true_false" else "Không" if q_type == "yes_no" else "Không có đủ thông tin"
        user_content = (
            f"Đồ thị: {graph_context}\n\n"
            f"Câu hỏi: {question_text}\n\n"
            f"Trả lời (bắt đầu bằng '{answer_keyword}'):\n"
        )
    else:
        user_content = (
            f"Đồ thị: {graph_context}\n\n"
            f"Câu hỏi: {question_text}\n\n"
            f"Trả lời (bắt đầu bằng 'Đúng' hoặc 'Sai'):\n"
        )

    return {"system_prompt": system_prompt, "user_content": user_content}


def retrieve_paths_for_question(
    question_text: str,
    hop_hint: int,
    store: GraphStore,
    max_paths: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve candidate multi-hop paths from Neo4j for a free-form question.

    This function uses a three-tier approach:
    1. Primary: NER to extract entity names from the question
    2. Fallback: Alias linker to catch entities NER might miss
    3. Additional: Neo4j fuzzy text search

    Args:
        question_text: The question text to retrieve paths for.
        hop_hint: Expected number of hops (2, 3, or 4). Clamped to [2, 4].
        store: GraphStore instance for Neo4j queries.
        max_paths: Maximum number of paths to return.

    Returns:
        List of path dictionaries, each with 'hop_count', 'path', and 'relationships' keys.
    """
    hop_hint = max(2, min(4, int(hop_hint)))

    # Step 1: Extract entities using NER (primary method)
    ids_from_ner = []
    ner_person_names = []
    ner_event_names = []

    try:
        from chatbot.alias_linker import _load_alias_index
        from chatbot.ner_extractor import extract_entities, extract_event_names, extract_person_names

        ner_person_names = extract_person_names(question_text)
        ner_event_names = extract_event_names(question_text)
        ner_entities = extract_entities(question_text)
        alias_index = _load_alias_index()

        # Link person names to entity IDs
        for name in ner_person_names:
            if name and len(name) > 2:
                normalized_name = " ".join(name.lower().strip().split())

                # Search Neo4j first (more reliable)
                ner_nodes = store.find_nodes_in_text(text=name, labels=None, limit=10)
                for node in ner_nodes:
                    if node.id not in ids_from_ner:
                        ids_from_ner.append(int(node.id))

                # Also check alias index (prioritize longer/more specific matches)
                matching_aliases = []
                if normalized_name in alias_index:
                    matching_aliases.append((normalized_name, alias_index[normalized_name]))

                for alias_key, alias_eids in alias_index.items():
                    if normalized_name in alias_key and (alias_key, alias_eids) not in matching_aliases:
                        matching_aliases.append((alias_key, alias_eids))

                matching_aliases.sort(key=lambda x: len(x[0]), reverse=True)
                for alias_key, alias_eids in matching_aliases:
                    for eid in alias_eids:
                        if eid not in ids_from_ner:
                            ids_from_ner.append(eid)

        # Link event names to entity IDs
        for name in ner_event_names:
            if name and len(name) > 3:
                normalized_name = " ".join(name.lower().strip().split())
                if normalized_name in alias_index:
                    for eid in alias_index[normalized_name]:
                        if eid not in ids_from_ner:
                            ids_from_ner.append(eid)

                ner_nodes = store.find_nodes_in_text(text=name, labels=None, limit=5)
                for node in ner_nodes:
                    if node.id not in ids_from_ner:
                        ids_from_ner.append(int(node.id))

        # Process all other entities from NER
        for entity in ner_entities:
            word = entity.get("word", "").strip()
            if word and len(word) > 2:
                word = word.replace("##", "").replace("▁", " ").strip()
                normalized_name = " ".join(word.lower().strip().split())
                if normalized_name in alias_index:
                    for eid in alias_index[normalized_name]:
                        if eid not in ids_from_ner:
                            ids_from_ner.append(eid)
    except Exception as e:
        print(f"Warning: NER extraction failed: {e}")

    # Step 2: Use alias linker as fallback
    ids_from_alias = link_entities_in_text(question_text, loose_match=True)

    # Step 3: Use Neo4j fuzzy text search as additional fallback
    nodes_from_title = store.find_nodes_in_text(text=question_text, labels=None, limit=50)
    ids_from_title = [int(n.id) for n in nodes_from_title]

    # Combine all sources
    all_candidate_ids = sorted({*ids_from_ner, *ids_from_alias, *ids_from_title})
    event_ids = []

    # Classify entities as Person or Event based on Neo4j labels
    person_ids_from_ner = []
    person_ids_from_alias = []
    person_ids_from_title = []

    for pid in ids_from_ner:
        result = store._run("MATCH (n) WHERE n.id = $id RETURN labels(n) AS labels LIMIT 1", {"id": int(pid)})
        if result:
            labels = result[0].get("labels", [])
            if "Event" in labels:
                if pid not in event_ids:
                    event_ids.append(pid)
            else:
                person_ids_from_ner.append(pid)

    for pid in ids_from_alias:
        if pid not in ids_from_ner:
            result = store._run("MATCH (n) WHERE n.id = $id RETURN labels(n) AS labels LIMIT 1", {"id": int(pid)})
            if result:
                labels = result[0].get("labels", [])
                if "Event" in labels:
                    if pid not in event_ids:
                        event_ids.append(pid)
                else:
                    if pid not in person_ids_from_ner:
                        person_ids_from_alias.append(pid)

    for node in nodes_from_title:
        node_id = int(node.id)
        if node_id not in ids_from_ner and node_id not in ids_from_alias:
            if node.label == "Event":
                if node_id not in event_ids:
                    event_ids.append(node_id)
            else:
                person_ids_from_title.append(node_id)

    # Combine person IDs with priority: NER > alias > title
    primary_person_ids = person_ids_from_ner
    secondary_person_ids = person_ids_from_alias
    tertiary_person_ids = person_ids_from_title
    all_person_ids = (
        primary_person_ids
        + [p for p in secondary_person_ids if p not in primary_person_ids]
        + [p for p in tertiary_person_ids if p not in primary_person_ids and p not in secondary_person_ids]
    )

    # Identify most relevant person IDs (for disambiguation)
    most_relevant_person_ids = person_ids_from_ner[:5]

    if ner_person_names:
        try:
            from chatbot.alias_linker import _load_alias_index
            import re

            alias_index = _load_alias_index()
            question_lower = question_text.lower()
            best_matches = []
            other_matches = []

            for name in ner_person_names:
                name_lower = name.lower()
                for alias_key, alias_ids in alias_index.items():
                    if name_lower in alias_key or alias_key in name_lower:
                        has_disambiguation = False
                        if "(" in question_lower and "(" in alias_key:
                            q_markers = re.findall(r'\([^)]+\)', question_lower)
                            a_markers = re.findall(r'\([^)]+\)', alias_key)
                            if q_markers and a_markers:
                                for q_m in q_markers:
                                    for a_m in a_markers:
                                        if q_m.lower() == a_m.lower():
                                            has_disambiguation = True
                                            break

                        if not has_disambiguation:
                            name_pos = question_lower.find(name_lower)
                            if name_pos >= 0:
                                after_name = question_lower[name_pos + len(name_lower) : name_pos + len(name_lower) + 5].strip()
                                alias_suffix = alias_key[len(name_lower):].strip() if alias_key.startswith(name_lower) else ""
                                if alias_suffix and alias_suffix.lower() in after_name.lower():
                                    has_disambiguation = True

                        for eid in alias_ids:
                            if eid in person_ids_from_ner:
                                if has_disambiguation:
                                    if eid not in best_matches:
                                        best_matches.append(eid)
                                else:
                                    if eid not in other_matches and eid not in best_matches:
                                        other_matches.append(eid)

            if best_matches:
                most_relevant_person_ids = best_matches + [eid for eid in most_relevant_person_ids if eid not in best_matches]
        except Exception:
            pass

    if not all_candidate_ids:
        return []

    # Step 4: Query paths from Neo4j
    if hop_hint == 2:
        initial_max = max(max_paths * 15, 100)
    elif hop_hint == 3:
        initial_max = max(max_paths * 20, 150)
    else:
        initial_max = max(max_paths * 25, 200)

    if hop_hint == 2:
        person_ids_to_use = None
        if len(primary_person_ids) >= 2:
            person_ids_to_use = primary_person_ids
        elif len(all_person_ids) >= 2:
            person_ids_to_use = all_person_ids

        paths = store.sample_paths_for_candidates_2hop(all_candidate_ids, initial_max, person_ids=person_ids_to_use)
    elif hop_hint == 3:
        paths = store.sample_paths_for_candidates_3hop(
            all_candidate_ids, initial_max, person_ids=all_person_ids if all_person_ids else None
        )
    else:
        paths = store.sample_paths_for_candidates_4hop(all_candidate_ids, initial_max)

    # Step 5: Try broader queries if needed
    if all_person_ids and len(paths) < max_paths:
        additional_paths_2hop = store.sample_paths_for_candidates_2hop(
            all_candidate_ids + all_person_ids, initial_max // 2, person_ids=None
        )
        existing_path_keys = {tuple(sorted([n["id"] for n in p["path"]])) for p in paths}
        for p in additional_paths_2hop:
            path_key = tuple(sorted([n["id"] for n in p["path"]]))
            if path_key not in existing_path_keys:
                paths.append(p)
                existing_path_keys.add(path_key)

    # Step 6: Rank and filter paths
    if paths:

        def path_score(path_dict: Dict[str, Any]) -> int:
            """Score a path based on entity matches.

            Higher scores indicate better matches with entities mentioned in the question.

            Args:
                path_dict: Path dictionary with 'path' key containing nodes.

            Returns:
                Integer score (higher is better).
            """
            path_nodes = path_dict.get("path", [])
            path_all_ids = {node.get("id") for node in path_nodes}
            path_person_ids = {node.get("id") for node in path_nodes if node.get("type") == "person"}
            path_event_ids = {node.get("id") for node in path_nodes if node.get("type") == "event"}

            score = 0

            # Massive bonus for paths containing all most relevant persons
            if most_relevant_person_ids:
                most_relevant_matches = sum(1 for pid in most_relevant_person_ids if pid in path_person_ids)
                if most_relevant_matches == len(most_relevant_person_ids) and len(most_relevant_person_ids) >= 2:
                    score += 100000
                elif most_relevant_matches > 0:
                    score += most_relevant_matches * 10000

            # Count person matches
            if all_person_ids:
                primary_matches = sum(1 for pid in primary_person_ids if pid in path_person_ids)
                secondary_matches = sum(1 for pid in secondary_person_ids if pid in path_person_ids)

                if path_nodes and path_nodes[0].get("id") in primary_person_ids:
                    primary_matches += 100
                if path_nodes and path_nodes[-1].get("id") in primary_person_ids:
                    primary_matches += 50
                if len(path_person_ids & set(all_person_ids)) >= 2:
                    primary_matches += 50

                score += primary_matches * 1000 + secondary_matches * 100

            # Count event matches
            event_matches = sum(1 for eid in event_ids if eid in path_event_ids)
            score += event_matches * 50

            # Bonus for having both person and event matches
            if all_person_ids and primary_matches > 0 and event_matches > 0:
                score += 200

            # Check candidate ID matches
            candidate_matches = len(path_all_ids & set(all_candidate_ids))
            score += candidate_matches * 10

            # Small penalty if we have primary_person_ids but path doesn't contain any
            if all_person_ids and primary_person_ids:
                if not any(pid in path_person_ids for pid in primary_person_ids):
                    score = max(0, score - 100)

            return score

        paths.sort(key=path_score, reverse=True)

        # Filter paths by primary person matches
        if primary_person_ids:
            paths_with_both_primary = []
            paths_with_one_primary = []
            paths_without_primary = []

            for p in paths:
                path_person_ids = {node.get("id") for node in p.get("path", []) if node.get("type") == "person"}
                primary_count = sum(1 for pid in primary_person_ids if pid in path_person_ids)

                if primary_count >= 2:
                    paths_with_both_primary.append(p)
                elif primary_count >= 1:
                    paths_with_one_primary.append(p)
                else:
                    paths_without_primary.append(p)

            paths = paths_with_both_primary[: max_paths * 2]
            paths.extend(paths_with_one_primary[:max_paths])
            paths.extend(paths_without_primary[: max_paths // 2])
        else:
            paths = paths[: max_paths * 2]

        # Final ranking and selection
        paths.sort(key=path_score, reverse=True)
        paths = paths[:max_paths]

    return paths


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
