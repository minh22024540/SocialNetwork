"""
Alias-based entity linker using the precomputed JSON index.

We keep this separate from Neo4j so that:
- We can reuse the rich alias data from `entity_aliases_map.json`.
- We don't have to modify the Neo4j schema to add alias properties.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


ALIAS_INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "entity_aliases_map.json"

_NAME_TO_ENTITIES: Dict[str, List[int]] | None = None


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def _load_alias_index() -> Dict[str, List[int]]:
    """Load name_to_entities from JSON once and cache it."""
    global _NAME_TO_ENTITIES
    if _NAME_TO_ENTITIES is not None:
        return _NAME_TO_ENTITIES

    if not ALIAS_INDEX_PATH.exists():
        _NAME_TO_ENTITIES = {}
        return _NAME_TO_ENTITIES

    with ALIAS_INDEX_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    name_to_entities_raw = data.get("name_to_entities", {})

    # Keys in JSON are strings -> ensure entity IDs are ints.
    name_to_entities: Dict[str, List[int]] = {}
    for name, ids in name_to_entities_raw.items():
        if not isinstance(ids, list):
            continue
        norm_name = _normalize(name)
        if not norm_name:
            continue
        int_ids = []
        for eid in ids:
            try:
                int_ids.append(int(eid))
            except Exception:
                continue
        if int_ids:
            name_to_entities[norm_name] = int_ids

    _NAME_TO_ENTITIES = name_to_entities
    return _NAME_TO_ENTITIES


def _generate_ngrams(words: List[str], max_n: int = 5) -> Iterable[str]:
    """Generate contiguous n-grams (1..max_n) from a word list."""
    n_words = len(words)
    for n in range(1, max_n + 1):
        for i in range(0, max(0, n_words - n + 1)):
            yield " ".join(words[i : i + n])


def link_entities_in_text(text: str, max_ngrams: int = 5) -> List[int]:
    """Link entities in text using alias index (exact match on normalized n-grams).

    This is complementary to fuzzy title matching in Neo4j:
    - Here chúng ta dùng full title/label/alias đã chuẩn hóa.
    - Neo4j `find_nodes_in_text` xử lý thêm các case partial / typo nhẹ theo title.
    """
    index = _load_alias_index()
    if not index:
        return []

    norm = _normalize(text)
    if not norm:
        return []

    words = norm.split(" ")
    seen: Set[int] = set()

    for ngram in _generate_ngrams(words, max_n=max_ngrams):
        if ngram in index:
            for eid in index[ngram]:
                seen.add(eid)

    return sorted(seen)


__all__ = ["link_entities_in_text"]


