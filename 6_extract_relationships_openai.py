#!/usr/bin/env python3
"""
Extract typed relationships between person-event pairs using OpenAI API.

This module reads Wikipedia text and uses OpenAI to extract relationship types
between person and event entities, handling entity name aliases for fuzzy matching.
"""

import json
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    from openai import AsyncOpenAI
    from openai import RateLimitError, APIError
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    raise


# OpenAI API configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it with: export OPENAI_API_KEY='your-key-here'")

# Rate limiting and concurrency
MAX_CONCURRENT_REQUESTS = 30  # Maximum concurrent API requests
MAX_RETRIES = 5  # Maximum retry attempts
BATCH_SIZE = 50  # Process this many pairs before writing
RATE_LIMIT_BASE_DELAY = 1.0  # Base delay for exponential backoff (seconds)


def load_relationship_types(types_file: Path) -> List[Dict]:
    """Load relationship type definitions from JSON file.
    
    Args:
        types_file: Path to relationship_types.json.
        
    Returns:
        List of relationship type dictionaries.
    """
    if not types_file.exists():
        raise FileNotFoundError(f"Relationship types file not found: {types_file}")
    
    with types_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("relationship_types", [])


def load_entity_aliases(aliases_file: Path) -> Dict:
    """Load entity alias mapping from JSON file.
    
    Args:
        aliases_file: Path to entity_aliases_map.json.
        
    Returns:
        Dictionary with entity_map and name_to_entities.
    """
    if not aliases_file.exists():
        raise FileNotFoundError(f"Entity aliases file not found: {aliases_file}")
    
    with aliases_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_entity_mentions(text: str, entity_names: List[str]) -> List[Tuple[int, int, str]]:
    """Find all mentions of entity names in text.
    
    Args:
        text: Text to search in.
        entity_names: List of normalized entity names to find.
        
    Returns:
        List of tuples (start_pos, end_pos, matched_name) for each mention.
    """
    mentions = []
    text_lower = text.lower()
    
    for name in entity_names:
        if not name:
            continue
        # Escape special regex characters
        pattern = re.escape(name)
        # Find all occurrences
        for match in re.finditer(pattern, text_lower):
            mentions.append((match.start(), match.end(), name))
    
    # Sort by position
    mentions.sort(key=lambda x: x[0])
    return mentions


def extract_context_snippets(
    text: str,
    person_names: List[str],
    event_names: List[str],
    context_window: int = 1200,  # Larger window to capture indirect references
    max_distance: int = 2000  # Maximum distance between mentions to consider
) -> List[str]:
    """Extract text snippets where both person and event are mentioned.

    Extracts snippets using two strategies:
    1. Direct mentions: Both entities mentioned by name/alias in snippet
    2. Indirect references: One entity mentioned, extract context that may
       contain indirect references to the other (pronouns, titles, etc.)

    Args:
        text: Full Wikipedia text.
        person_names: List of normalized person entity names.
        event_names: List of normalized event entity names.
        context_window: Number of characters around mentions to include.
        max_distance: Maximum character distance between mentions to consider.
        
    Returns:
        List of text snippets containing both entities (directly or indirectly).
    """
    person_mentions = find_entity_mentions(text, person_names)
    event_mentions = find_entity_mentions(text, event_names)
    
    snippets = []
    text_len = len(text)
    
    # Strategy 1: Extract snippets where BOTH entities are directly mentioned
    if person_mentions and event_mentions:
        mention_pairs = []
        for p_start, p_end, _ in person_mentions:
            for e_start, e_end, _ in event_mentions:
                # Calculate distance between mentions
                distance = abs(p_start - e_start)
                if distance <= max_distance:  # Only consider nearby mentions
                    mention_pairs.append((distance, p_start, p_end, e_start, e_end))
        
        # Sort by distance (closer mentions first) to prioritize relevant snippets
        mention_pairs.sort(key=lambda x: x[0])
        
        for distance, p_start, p_end, e_start, e_end in mention_pairs:
            # Calculate snippet boundaries with context window
            if p_start < e_start:
                snippet_start = max(0, p_start - context_window // 2)
                snippet_end = min(text_len, e_end + context_window // 2)
            else:
                snippet_start = max(0, e_start - context_window // 2)
                snippet_end = min(text_len, p_end + context_window // 2)
            
            snippet = text[snippet_start:snippet_end].strip()
            if snippet and len(snippet) > 50:  # Minimum snippet length
                snippets.append(snippet)
    
    # Strategy 2: Extract larger context around entity mentions
    # This captures cases where one entity is mentioned directly but the other
    # might be referenced indirectly (pronouns, titles, etc.)
    # We'll let OpenAI figure out the relationship from the context
    
    # Extract context around person mentions (in case event is mentioned indirectly nearby)
    if person_mentions:
        for p_start, p_end, _ in person_mentions[:15]:  # Limit to avoid too many snippets
            snippet_start = max(0, p_start - context_window)
            snippet_end = min(text_len, p_end + context_window)
            snippet = text[snippet_start:snippet_end].strip()
            if snippet and len(snippet) > 100:  # Larger minimum for context snippets
                snippets.append(snippet)
    
    # Extract context around event mentions (in case person is mentioned indirectly nearby)
    if event_mentions:
        for e_start, e_end, _ in event_mentions[:15]:  # Limit to avoid too many snippets
            snippet_start = max(0, e_start - context_window)
            snippet_end = min(text_len, e_end + context_window)
            snippet = text[snippet_start:snippet_end].strip()
            if snippet and len(snippet) > 100:  # Larger minimum for context snippets
                snippets.append(snippet)
    
    # Deduplicate while preserving order
    seen = set()
    unique_snippets = []
    for snippet in snippets:
        snippet_hash = hash(snippet[:150])  # Hash first 150 chars for dedup
        if snippet_hash not in seen:
            seen.add(snippet_hash)
            unique_snippets.append(snippet)
    
    # Return up to 15 snippets to capture all relationship mentions
    # This includes both direct mentions and indirect references
    return unique_snippets[:15]


async def extract_relationship_openai_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    person_name: str,
    event_name: str,
    text_snippets: List[str],
    relationship_types: List[Dict]
) -> Optional[Dict]:
    """Extract relationship type using OpenAI API (async version with rate limiting).
    
    Args:
        client: AsyncOpenAI client instance.
        semaphore: Semaphore to limit concurrent requests.
        person_name: Person entity name.
        event_name: Event entity name.
        text_snippets: List of text snippets mentioning both entities.
        relationship_types: List of available relationship type definitions.
        
    Returns:
        Dictionary with type, confidence, and evidence, or None if extraction fails.
    """
    if not text_snippets:
        return None
    
    # Build relationship type list for prompt
    type_list = "\n".join([
        f"- {rt['type']}: {rt['description']}"
        for rt in relationship_types
    ])
    
    # Combine snippets - use more snippets to capture all relationship mentions
    # This includes both direct mentions and indirect references (pronouns, titles, etc.)
    # Use up to 12 snippets to ensure we capture all relationship information
    max_snippets = min(12, len(text_snippets))
    combined_text = "\n\n---\n\n".join(text_snippets[:max_snippets])
    
    prompt = f"""Extract the relationship type between the person "{person_name}" and the event "{event_name}" from the following text snippets.

The text snippets below are extracted from Wikipedia articles. Note that:
- The person or event might be mentioned directly by name/alias
- OR they might be referenced indirectly (using pronouns like "he", "she", "they", titles like "the general", "the leader", or other indirect references)
- The relationship might be described in multiple sentences or paragraphs

Analyze ALL snippets carefully to identify the relationship(s) between them, even if one entity is mentioned indirectly.

Available relationship types:
{type_list}

If none of these types fit, suggest a new appropriate type name.

Analyze the text and return a JSON object with:
- "type": the relationship type (use existing type or suggest new one)
- "confidence": a float between 0.0 and 1.0 indicating confidence
- "evidence": a short quote from the text that supports this relationship

Text snippets (separated by "---"):
{combined_text}

Return only valid JSON, no other text."""

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a relationship extraction expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    # max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Validate result
                    if "type" in result and "confidence" in result:
                        return {
                            "type": result["type"].upper().replace(" ", "_"),
                            "confidence": float(result.get("confidence", 0.5)),
                            "evidence": result.get("evidence", text_snippets[0][:200])
                        }
                
                # If no valid JSON found, return default
                return {
                    "type": "RELATED_TO",
                    "confidence": 0.3,
                    "evidence": text_snippets[0][:200] if text_snippets else ""
                }
                
            except json.JSONDecodeError:
                # JSON parsing error - retry with backoff
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return None
            except RateLimitError as e:
                # Rate limit error - use exponential backoff
                backoff_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff_time)
                    continue
                # Final attempt failed
                return None
            except APIError as e:
                # Check if it's a rate limit by status code
                is_rate_limit = False
                if hasattr(e, 'status_code') and e.status_code == 429:
                    is_rate_limit = True
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if e.response.status_code == 429:
                        is_rate_limit = True
                
                if is_rate_limit:
                    # Exponential backoff for rate limits
                    backoff_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(backoff_time)
                        continue
                elif attempt < MAX_RETRIES - 1:
                    # Regular exponential backoff for other API errors
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return None
            except Exception as e:
                # Check for rate limit in error message or attributes
                error_str = str(e)
                is_rate_limit = False
                
                # Check status code
                if hasattr(e, 'status_code') and e.status_code == 429:
                    is_rate_limit = True
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if e.response.status_code == 429:
                        is_rate_limit = True
                # Check error message
                elif "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                    is_rate_limit = True
                
                if is_rate_limit:
                    # Exponential backoff for rate limits
                    backoff_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(backoff_time)
                        continue
                elif attempt < MAX_RETRIES - 1:
                    # Regular exponential backoff for other errors
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                # Don't print every error to avoid spam
                if attempt == MAX_RETRIES - 1:
                    print(f"OpenAI API error (final attempt): {type(e).__name__}: {error_str[:100]}")
                return None
        
        return None


async def process_entity_pair_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    person_id: int,
    event_id: int,
    person_data: Dict,
    event_data: Dict,
    person_text: str,
    event_text: str,
    alias_map: Dict,
    relationship_types: List[Dict]
) -> Optional[Dict]:
    """Process a single person-event pair to extract relationship.
    
    Args:
        client: OpenAI client instance.
        person_id: Person entity ID.
        event_id: Event entity ID.
        person_data: Person entity data from alias map.
        event_data: Event entity data from alias map.
        person_text: Person's Wikipedia article text.
        event_text: Event's Wikipedia article text.
        alias_map: Entity alias mapping.
        relationship_types: Relationship type definitions.
        
    Returns:
        Relationship dictionary or None if extraction fails.
    """
    # Get all name variants for person and event
    person_names = []
    if person_data.get("title"):
        person_names.append(person_data["title"].lower())
    for lang in ["vi", "en"]:
        if lang in person_data.get("labels", {}):
            person_names.append(person_data["labels"][lang].lower())
        for alias in person_data.get("aliases", {}).get(lang, []):
            person_names.append(alias.lower())
    
    event_names = []
    if event_data.get("title"):
        event_names.append(event_data["title"].lower())
    for lang in ["vi", "en"]:
        if lang in event_data.get("labels", {}):
            event_names.append(event_data["labels"][lang].lower())
        for alias in event_data.get("aliases", {}).get(lang, []):
            event_names.append(alias.lower())
    
    # Extract snippets from both texts
    snippets_person = extract_context_snippets(person_text, person_names, event_names)
    snippets_event = extract_context_snippets(event_text, person_names, event_names)
    
    all_snippets = snippets_person + snippets_event
    if not all_snippets:
        return None
    
    # Use primary names for API call
    person_primary = person_data.get("title") or person_data.get("labels", {}).get("vi") or person_data.get("labels", {}).get("en", "Person")
    event_primary = event_data.get("title") or event_data.get("labels", {}).get("vi") or event_data.get("labels", {}).get("en", "Event")
    
    # Extract relationship using async function
    result = await extract_relationship_openai_async(
        client,
        semaphore,
        person_primary,
        event_primary,
        all_snippets,
        relationship_types
    )
    
    if not result:
        return None
    
    return {
        "source_id": person_id,
        "target_id": event_id,
        "relationship_type": result["type"],
        "confidence": result["confidence"],
        "evidence_text": result["evidence"],
        "source": "wikipedia"
    }


async def process_pairs_async(
    pairs_to_process: List[Tuple[int, int, Dict, Dict, str, str]],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    alias_data: Dict,
    relationship_types: List[Dict],
    stats: Dict,
    pbar: tqdm
) -> List[Dict]:
    """Process pairs concurrently using async.
    
    Args:
        pairs_to_process: List of tuples (person_id, event_id, person_data, event_data, person_text, event_text).
        client: AsyncOpenAI client.
        semaphore: Semaphore for rate limiting.
        alias_data: Entity alias mapping.
        relationship_types: Relationship type definitions.
        stats: Statistics dictionary to update.
        pbar: Progress bar.
        
    Returns:
        List of extracted relationships.
    """
    async def process_with_tracking(person_id, event_id, person_data, event_data, person_text, event_text):
        """Wrapper to track progress."""
        try:
            result = await process_entity_pair_async(
                client,
                semaphore,
                person_id,
                event_id,
                person_data,
                event_data,
                person_text,
                event_text,
                alias_data,
                relationship_types
            )
            stats["processed"] += 1
            pbar.update(1)
            
            if result:
                stats["extracted"] += 1
                return result
            else:
                stats["failed"] += 1
                return None
        except Exception as e:
            stats["failed"] += 1
            stats["processed"] += 1
            pbar.update(1)
            return None
    
    # Create all tasks
    tasks = [
        process_with_tracking(person_id, event_id, person_data, event_data, person_text, event_text)
        for person_id, event_id, person_data, event_data, person_text, event_text in pairs_to_process
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None and exceptions
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    
    return valid_results


async def main_async():
    """Async main function to extract relationships with concurrent processing."""
    # Try main directory first, then old_code
    project_root = Path(__file__).resolve().parents[2]
    data_raw = Path(__file__).resolve().parent / "data_raw"
    input_file = data_raw / "wiki_entities_with_links.jsonl"
    if not input_file.exists():
        input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"
    types_file = Path(__file__).resolve().parent / "data" / "6_relationship_types.json"
    aliases_file = Path(__file__).resolve().parent / "data" / "entity_aliases_map.json"
    output_file = data_raw / "enriched_relationships.jsonl"
    cache_file = data_raw / "enriched_relationships_cache.json"
    
    print("=" * 80)
    print("Extracting Relationships with OpenAI (Concurrent Mode)")
    print("=" * 80)
    
    # Load relationship types
    relationship_types = load_relationship_types(types_file)
    print(f"Loaded {len(relationship_types)} relationship types")
    
    # Load entity aliases
    alias_data = load_entity_aliases(aliases_file)
    entity_map = alias_data["entity_map"]
    print(f"Loaded {len(entity_map)} entities with aliases")
    
    # Load cache of already processed pairs
    processed_pairs = set()
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            cache_data = json.load(f)
            processed_pairs = set(tuple(p) for p in cache_data.get("processed_pairs", []))
        print(f"Loaded {len(processed_pairs)} cached pairs")
    
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    print(f"Using OpenAI API:")
    print(f"  Model: {OPENAI_MODEL}")
    print(f"  Base URL: {OPENAI_BASE_URL}")
    print(f"  Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"  API Key: {'*' * 20}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 4 else '****'}")
    
    # Read input JSONL and process pairs
    stats = {
        "total_pairs": 0,
        "processed": 0,
        "extracted": 0,
        "failed": 0,
        "cached": len(processed_pairs)
    }
    
    # Build entity lookup by ID (convert string keys to int for consistency)
    entity_lookup = {}
    for entity_id_str, entity_data in entity_map.items():
        try:
            entity_id = int(entity_id_str)
            entity_lookup[entity_id] = entity_data
        except (ValueError, TypeError):
            continue
    
    # Read all entities first to build text lookup
    entity_texts = {}
    print("Loading entity texts...")
    with input_file.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading texts", unit="entity"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                wp = obj.get("wikipedia", {})
                entity_id = wp.get("id")
                if isinstance(entity_id, int):
                    # Get full text
                    text = wp.get("full_text") or wp.get("text") or wp.get("content") or ""
                    if text:
                        entity_texts[entity_id] = text
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded texts for {len(entity_texts)} entities")
    
    # Collect all pairs to process
    print("\nCollecting pairs to process...")
    pairs_to_process = []
    new_processed_pairs = []
    
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            wp = obj.get("wikipedia", {})
            entity_id = wp.get("id")
            category = obj.get("category")
            links = obj.get("links", [])
            
            if not isinstance(entity_id, int) or category != "person":
                continue
            
            # Process each link to an event
            for target_id in links:
                if not isinstance(target_id, int):
                    continue
                
                pair = (entity_id, target_id)
                stats["total_pairs"] += 1
                
                # Skip if already processed
                if pair in processed_pairs:
                    continue
                
                # Check if target is an event
                target_data = entity_lookup.get(target_id)
                if not target_data or target_data.get("category") != "event":
                    continue
                
                person_data = entity_lookup.get(entity_id)
                if not person_data:
                    continue
                
                # Get texts
                person_text = entity_texts.get(entity_id, "")
                event_text = entity_texts.get(target_id, "")
                
                if not person_text or not event_text:
                    continue
                
                # Add to processing queue
                pairs_to_process.append((entity_id, target_id, person_data, target_data, person_text, event_text))
                new_processed_pairs.append([entity_id, target_id])
    
    print(f"Found {len(pairs_to_process)} pairs to process")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Process pairs in batches concurrently
    print("\nExtracting relationships (concurrent mode)...")
    all_results = []
    
    with tqdm(total=len(pairs_to_process), desc="Processing pairs", unit="pair") as pbar:
        # Process in batches to manage memory and write results periodically
        for i in range(0, len(pairs_to_process), BATCH_SIZE):
            batch = pairs_to_process[i:i + BATCH_SIZE]
            batch_results = await process_pairs_async(
                batch,
                client,
                semaphore,
                alias_data,
                relationship_types,
                stats,
                pbar
            )
            
            all_results.extend(batch_results)
            
            # Write batch results
            if batch_results:
                with output_file.open("a", encoding="utf-8") as outf:
                    for res in batch_results:
                        outf.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    # Update cache
    all_processed = list(processed_pairs) + new_processed_pairs
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump({"processed_pairs": all_processed}, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"Total person-event pairs: {stats['total_pairs']:,}")
    print(f"Already cached: {stats['cached']:,}")
    print(f"Newly processed: {stats['processed']:,}")
    print(f"Successfully extracted: {stats['extracted']:,}")
    print(f"Failed extractions: {stats['failed']:,}")
    print(f"\nOutput saved to: {output_file}")


def main():
    """Main function wrapper to run async code."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

