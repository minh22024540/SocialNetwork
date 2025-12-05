"""Alias-based entity linker using a precomputed JSON index.

This module provides entity linking functionality using the entity_aliases_map.json
file. It's kept separate from Neo4j to:
- Reuse rich alias data from the JSON file
- Avoid modifying the Neo4j schema to add alias properties
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

ALIAS_INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "entity_aliases_map.json"

_NAME_TO_ENTITIES: Dict[str, List[int]] | None = None


def _normalize(text: str) -> str:
    """Normalize text for matching.

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text (lowercase, extra spaces removed, parentheses handled).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace("(", " ").replace(")", " ")
    return " ".join(text.split())


def _load_alias_index() -> Dict[str, List[int]]:
    """Load name-to-entities mapping from JSON file and cache it.

    Returns:
        Dictionary mapping normalized names to lists of entity IDs.
    """
    global _NAME_TO_ENTITIES
    if _NAME_TO_ENTITIES is not None:
        return _NAME_TO_ENTITIES

    if not ALIAS_INDEX_PATH.exists():
        _NAME_TO_ENTITIES = {}
        return _NAME_TO_ENTITIES

    with ALIAS_INDEX_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    name_to_entities_raw = data.get("name_to_entities", {})

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
    """Generate contiguous n-grams (1..max_n) from a word list.

    Args:
        words: List of words to generate n-grams from.
        max_n: Maximum n-gram size.

    Yields:
        N-gram strings of size 1 to max_n.
    """
    n_words = len(words)
    for n in range(1, max_n + 1):
        for i in range(0, max(0, n_words - n + 1)):
            yield " ".join(words[i : i + n])


def link_entities_in_text(text: str, max_ngrams: int = 5, loose_match: bool = True) -> List[int]:
    """Link entities in text using the alias index.

    Uses exact n-gram matching and optionally loose phrase/substring matching.
    This is complementary to fuzzy title matching in Neo4j.

    Args:
        text: Input text to extract entities from.
        max_ngrams: Maximum n-gram size for exact matching.
        loose_match: If True, also performs partial/substring matching on aliases.

    Returns:
        Sorted list of entity IDs found in the text.
    """
    index = _load_alias_index()
    if not index:
        return []

    norm = _normalize(text)
    if not norm:
        return []

    words = norm.split(" ")
    seen: Set[int] = set()

    # Prioritize longer, more specific aliases first (for disambiguation)
    if loose_match:
        sorted_aliases = sorted(index.items(), key=lambda x: len(x[0]), reverse=True)
        for alias, eids in sorted_aliases:
            if alias in norm:
                for eid in eids:
                    seen.add(eid)

    # Exact n-gram matching
    for ngram in _generate_ngrams(words, max_n=max_ngrams):
        if ngram in index:
            for eid in index[ngram]:
                seen.add(eid)

    # Loose matching: check if alias appears as a contiguous phrase
    if loose_match:
        for alias, eids in index.items():
            if len(alias) < 3:
                continue

            if alias in norm:
                for eid in eids:
                    seen.add(eid)
            else:
                # Check if words from alias appear in text (allowing reordering)
                alias_words = alias.split(" ")
                if len(alias_words) >= 2:
                    common_words = {"nhà", "văn", "diễn", "viên", "ông", "bà", "cô", "anh", "chị"}
                    significant_alias_words = [w for w in alias_words if len(w) > 2 and w not in common_words]

                    if len(significant_alias_words) >= 2:
                        words_in_text = set(words)
                        matching_words = sum(1 for aw in significant_alias_words if aw in words_in_text)

                        if matching_words == len(significant_alias_words):
                            significant_positions = [i for i, w in enumerate(words) if w in significant_alias_words]
                            if len(significant_positions) >= 2:
                                max_gap = max(significant_positions) - min(significant_positions)
                                if max_gap <= 10:
                                    for eid in eids:
                                        seen.add(eid)
                    elif len(significant_alias_words) == 1 and len(alias_words) >= 2:
                        if significant_alias_words[0] in words:
                            other_words = [w for w in alias_words if w != significant_alias_words[0] and len(w) > 1]
                            if any(w in words for w in other_words):
                                for eid in eids:
                                    seen.add(eid)
                    elif len(alias_words) >= 2:
                        words_in_text = set(words)
                        matching_words = sum(1 for aw in alias_words if aw in words_in_text)
                        if matching_words >= 2:
                            for eid in eids:
                                seen.add(eid)

    return sorted(seen)


__all__ = ["link_entities_in_text"]
