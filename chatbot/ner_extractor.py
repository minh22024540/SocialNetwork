"""Named Entity Recognition (NER) module for extracting entities from Vietnamese text.

This module uses HuggingFace Transformers with a multilingual NER model to extract
person names, event names, and other entities from Vietnamese questions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

# Multilingual NER model that works well with Vietnamese
MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

# Global pipeline cache
_ner_pipeline: Optional[Any] = None


def _get_ner_pipeline() -> Any:
    """Lazy load and cache the NER pipeline.

    Returns:
        The NER pipeline object, or None if loading fails.

    Raises:
        Prints warnings but does not raise exceptions.
    """
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline

    device = 0 if torch.cuda.is_available() else -1

    try:
        _ner_pipeline = pipeline(
            "ner",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            aggregation_strategy="simple",
            device=device,
        )
        print(f"Loaded NER model: {MODEL_NAME} on device: {'cuda' if device >= 0 else 'cpu'}")
    except Exception as e:
        print(f"Warning: Could not load NER model {MODEL_NAME}: {e}")
        print("Falling back to English-only model (note: poor performance on Vietnamese)...")
        try:
            _ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=device,
            )
            print("Loaded fallback NER model: dslim/bert-base-NER (limited Vietnamese support)")
        except Exception as e2:
            print(f"Error loading fallback NER model: {e2}")
            _ner_pipeline = None

    return _ner_pipeline


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract all named entities from text using the NER model.

    Args:
        text: Input text to extract entities from.

    Returns:
        List of entity dictionaries, each containing:
            - word: The entity text
            - entity_group: The entity type (PER, ORG, LOC, MISC, etc.)
            - score: Confidence score
            - start: Start position in text
            - end: End position in text
    """
    pipeline_obj = _get_ner_pipeline()
    if pipeline_obj is None:
        return []

    try:
        results = pipeline_obj(text)
        relevant_types = {"PER", "ORG", "LOC", "MISC", "PERSON", "ORGANIZATION", "LOCATION"}

        entities = []
        for result in results:
            entity_group = result.get("entity_group", "").upper()
            # Handle BIO tagging scheme (B-PER, I-PER, etc.)
            if entity_group.startswith(("B-", "I-")):
                entity_group = entity_group[2:]

            word = result.get("word", "").strip()

            is_relevant = (
                entity_group in relevant_types
                or entity_group.startswith("PER")
                or entity_group.startswith("ORG")
                or entity_group.startswith("LOC")
                or entity_group.startswith("MISC")
                or (not entity_group and word)
            )

            if is_relevant:
                entities.append({
                    "word": word,
                    "entity_group": entity_group,
                    "score": result.get("score", 0.0),
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                })

        return entities
    except Exception as e:
        print(f"Error in NER extraction: {e}")
        return []


def extract_person_names(text: str) -> List[str]:
    """Extract person names from text using NER.

    Args:
        text: Input text to extract person names from.

    Returns:
        List of person name strings, with Vietnamese prefixes removed.
    """
    entities = extract_entities(text)
    person_names = []

    for entity in entities:
        entity_group = entity.get("entity_group", "").upper()
        word = entity.get("word", "").strip()

        if entity_group in {"PER", "PERSON"} or entity_group.startswith("PER") or (not entity_group and word):
            word = word.replace("##", "").replace("▁", " ").strip()

            # Remove common Vietnamese prefixes
            prefixes = ["ông", "bà", "nhà văn", "diễn viên", "tướng", "quân nhân", "thiếu tướng", "đại tá"]
            for prefix in prefixes:
                if word.lower().startswith(prefix + " "):
                    word = word[len(prefix) + 1:].strip()

            if word and len(word) > 2:
                person_names.append(word)

    return person_names


def extract_all_entity_names(text: str) -> List[str]:
    """Extract all entity names (persons, organizations, locations, events) from text.

    Args:
        text: Input text to extract entities from.

    Returns:
        List of entity name strings, with Vietnamese prefixes removed.
    """
    entities = extract_entities(text)
    names = []

    for entity in entities:
        entity_group = entity.get("entity_group", "").upper()
        word = entity.get("word", "").strip()

        if word:
            word = word.replace("##", "").replace("▁", " ").strip()

            # Remove common Vietnamese prefixes
            prefixes = ["ông", "bà", "nhà văn", "diễn viên", "tướng", "quân nhân", "thiếu tướng", "đại tá"]
            for prefix in prefixes:
                if word.lower().startswith(prefix + " "):
                    word = word[len(prefix) + 1:].strip()

            if word and len(word) > 2:
                names.append(word)

    return names


def extract_event_names(text: str) -> List[str]:
    """Extract event names from text using NER.

    Events are typically labeled as MISC, LOC, or ORG, or contain event keywords
    like "chiến tranh", "sự kiện", etc.

    Args:
        text: Input text to extract event names from.

    Returns:
        List of event name strings.
    """
    entities = extract_entities(text)
    event_names = []

    for entity in entities:
        entity_group = entity.get("entity_group", "").upper()
        word = entity.get("word", "").strip()

        is_event = (
            entity_group in {"MISC", "LOC", "ORG"}
            or "chiến tranh" in word.lower()
            or "sự kiện" in word.lower()
            or "chiến dịch" in word.lower()
            or "hiệp định" in word.lower()
        )

        if is_event and word:
            word = word.replace("##", "").replace("▁", " ").strip()
            if word and len(word) > 3:
                event_names.append(word)

    return event_names
