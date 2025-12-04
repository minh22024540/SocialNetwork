#!/usr/bin/env python3
"""
Calculate edge weights based on mention frequency in Wikipedia text.

This module counts how many times each entity is mentioned in the other entity's
article and normalizes weights to represent connection strength.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm


def normalize_name(name: str) -> str:
    """Normalize entity name for matching.

    Converts name to lowercase and strips whitespace for consistent comparison.

    Args:
        name: Entity name to normalize. Can be any type.

    Returns:
        Normalized name string (lowercase, stripped). Returns empty string
        if input is not a string.
    """
    if not isinstance(name, str):
        return ""
    return name.lower().strip()


def get_entity_name_variants(entity_data: Dict) -> Set[str]:
    """Extract all name variants for an entity.

    Collects title, labels (preferring Vietnamese and English), and aliases
    (preferring Vietnamese and English) and normalizes them.

    Args:
        entity_data: Entity data dictionary containing:
            - title: str
            - labels: Dict[str, str] (lang -> label)
            - aliases: Dict[str, List[str]] (lang -> [alias1, ...])

    Returns:
        Set of normalized name variant strings (lowercase, stripped).
    """
    variants = set()
    
    # Add title
    title = entity_data.get("title", "")
    if title:
        variants.add(normalize_name(title))
    
    # Add labels (prefer Vietnamese and English)
    labels = entity_data.get("labels", {})
    for lang in ["vi", "en"]:
        if lang in labels:
            label = labels[lang]
            if label:
                variants.add(normalize_name(label))
    
    # Add aliases (prefer Vietnamese and English)
    aliases = entity_data.get("aliases", {})
    for lang in ["vi", "en"]:
        if lang in aliases:
            for alias in aliases[lang]:
                if alias:
                    variants.add(normalize_name(alias))
    
    return variants


def count_mentions(text: str, name_variants: Set[str]) -> int:
    """Count how many times any name variant appears in text.

    Performs case-insensitive substring matching using regex (with escaped
    special characters) to count occurrences of each variant.

    Args:
        text: Text to search in. Can be empty.
        name_variants: Set of normalized name variant strings to find.

    Returns:
        Total count of all mentions across all variants. Returns 0 if text
        or name_variants is empty.
    """
    if not text or not name_variants:
        return 0
    
    text_lower = text.lower()
    count = 0
    
    for variant in name_variants:
        if not variant:
            continue
        # Escape special regex characters
        pattern = re.escape(variant)
        # Count occurrences
        count += len(re.findall(pattern, text_lower))
    
    return count


def calculate_edge_weights(
    jsonl_path: Path,
    aliases_path: Path
) -> Dict[Tuple[int, int], float]:
    """Calculate edge weights for all entity pairs.

    Computes edge weights based on mention frequency: counts how many times
    each entity's name variants appear in the other entity's Wikipedia article
    text. Weights are normalized and represent connection strength.

    Args:
        jsonl_path: Path to wiki_entities_with_links.jsonl file containing
            entity data with full_text fields.
        aliases_path: Path to entity_aliases_map.json containing name variants.

    Returns:
        Dictionary mapping (source_id, target_id) tuples to normalized
        weight values (float). Weights are symmetric: (a, b) == (b, a).

    Raises:
        FileNotFoundError: If aliases_path does not exist.
    """
    # Load entity aliases
    if not aliases_path.exists():
        raise FileNotFoundError(f"Aliases file not found: {aliases_path}")
    
    with aliases_path.open("r", encoding="utf-8") as f:
        alias_data = json.load(f)
    
    entity_map = alias_data["entity_map"]
    print(f"Loaded {len(entity_map)} entities")
    
    # Build name variants for each entity
    # Note: entity_map keys might be strings, convert to int for consistency
    entity_variants = {}
    for entity_id_str, entity_data in entity_map.items():
        try:
            entity_id = int(entity_id_str)
        except (ValueError, TypeError):
            continue
        entity_variants[entity_id] = get_entity_name_variants(entity_data)
    
    # Load entity texts
    entity_texts = {}
    print("Loading entity texts...")
    with jsonl_path.open("r", encoding="utf-8") as f:
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
    
    # Calculate weights for all pairs
    edge_weights: Dict[Tuple[int, int], float] = {}
    mention_counts: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    print("Calculating edge weights...")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing edges", unit="entity"):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            wp = obj.get("wikipedia", {})
            source_id = wp.get("id")
            links = obj.get("links", [])
            
            if not isinstance(source_id, int):
                continue
            
            source_text = entity_texts.get(source_id, "")
            source_variants = entity_variants.get(source_id, set())
            
            if not source_text or not source_variants:
                continue
            
            # Process each link (can be list of IDs or list of dicts with target_id)
            for link_item in links:
                # Handle both formats: direct ID or dict with target_id
                if isinstance(link_item, dict):
                    target_id = link_item.get("target_id")
                elif isinstance(link_item, int):
                    target_id = link_item
                else:
                    continue
                
                if not isinstance(target_id, int):
                    continue
                
                # Create canonical pair (smaller ID first for undirected graph)
                pair = (min(source_id, target_id), max(source_id, target_id))
                
                if pair in mention_counts:
                    continue  # Already processed
                
                target_text = entity_texts.get(target_id, "")
                target_variants = entity_variants.get(target_id, set())
                
                if not target_text or not target_variants:
                    continue
                
                # Count mentions in both directions
                # How many times target is mentioned in source's article
                mentions_source_to_target = count_mentions(source_text, target_variants)
                
                # How many times source is mentioned in target's article
                mentions_target_to_source = count_mentions(target_text, source_variants)
                
                # Store raw counts
                mention_counts[pair] = (mentions_source_to_target, mentions_target_to_source)
    
    # Normalize weights
    print(f"\nMention counts found: {len(mention_counts)}")
    if not mention_counts:
        print("Warning: No mention counts found. Check if entity texts and variants are loaded correctly.")
        return edge_weights
    
    # Find max mentions for normalization
    all_mentions = []
    for (src_to_tgt, tgt_to_src) in mention_counts.values():
        all_mentions.append(src_to_tgt)
        all_mentions.append(tgt_to_src)
    
    max_mentions = max(all_mentions) if all_mentions else 1
    
    # Calculate normalized weights (0.0-1.0 scale)
    # Weight is average of bidirectional mentions, normalized
    for pair, (src_to_tgt, tgt_to_src) in mention_counts.items():
        total_mentions = src_to_tgt + tgt_to_src
        # Normalize: use logarithmic scaling for better distribution
        if total_mentions > 0:
            # Log scale: log(1 + mentions) / log(1 + max_mentions)
            import math
            normalized = math.log(1 + total_mentions) / math.log(1 + max_mentions)
            # Ensure range is 0.0-1.0
            weight = min(1.0, max(0.0, normalized))
        else:
            weight = 0.0
        
        edge_weights[pair] = weight
    
    return edge_weights


def update_jsonl_with_weights(
    input_path: Path,
    output_path: Path,
    edge_weights: Dict[Tuple[int, int], float]
) -> None:
    """Update JSONL file with weight information.
    
    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file with weights.
        edge_weights: Dictionary mapping (source_id, target_id) to weight.
    """
    print("Updating JSONL with weights...")
    
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="Writing weights", unit="entity"):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            wp = obj.get("wikipedia", {})
            source_id = wp.get("id")
            links = obj.get("links", [])
            
            if not isinstance(source_id, int):
                fout.write(line + "\n")
                continue
            
            # Add weight information to links
            weighted_links = []
            for link_item in links:
                # Handle both formats: direct ID or dict with target_id
                if isinstance(link_item, dict):
                    target_id = link_item.get("target_id")
                elif isinstance(link_item, int):
                    target_id = link_item
                else:
                    continue
                
                if not isinstance(target_id, int):
                    continue
                
                # Get canonical pair
                pair = (min(source_id, target_id), max(source_id, target_id))
                weight = edge_weights.get(pair, 1.0)  # Default weight 1.0 if not found
                
                weighted_links.append({
                    "target_id": target_id,
                    "weight": weight
                })
            
            # Add weights array to object
            obj["link_weights"] = weighted_links
            
            # Write updated object
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    """Main function to calculate edge weights."""
    # Try main directory first, then old_code
    project_root = Path(__file__).resolve().parents[2]
    data_raw = Path(__file__).resolve().parent / "data_raw"
    data_analysis = Path(__file__).resolve().parent / "data_analysis"
    input_file = data_raw / "wiki_entities_with_links.jsonl"
    if not input_file.exists():
        input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"
    aliases_file = Path(__file__).resolve().parent / "data" / "entity_aliases_map.json"
    output_file = data_raw / "wiki_entities_with_weights.jsonl"
    weights_file = data_analysis / "edge_weights.json"
    
    print("=" * 80)
    print("Calculating Edge Weights")
    print("=" * 80)
    
    try:
        # Calculate weights
        edge_weights = calculate_edge_weights(input_file, aliases_file)
        
        # Save weights to JSON
        weights_dict = {
            f"{pair[0]}_{pair[1]}": weight
            for pair, weight in edge_weights.items()
        }
        with weights_file.open("w", encoding="utf-8") as f:
            json.dump(weights_dict, f, indent=2)
        
        # Update JSONL with weights
        update_jsonl_with_weights(input_file, output_file, edge_weights)
        
        print("\n" + "=" * 80)
        print("Statistics")
        print("=" * 80)
        print(f"Total edges with weights: {len(edge_weights):,}")
        
        if edge_weights:
            weights_list = list(edge_weights.values())
            import statistics
            print(f"Average weight: {statistics.mean(weights_list):.4f}")
            print(f"Median weight: {statistics.median(weights_list):.4f}")
            print(f"Min weight: {min(weights_list):.4f}")
            print(f"Max weight: {max(weights_list):.4f}")
        
        print(f"\nWeights saved to: {weights_file}")
        print(f"Updated JSONL saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error calculating weights: {e}")
        raise


if __name__ == "__main__":
    main()

