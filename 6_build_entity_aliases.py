#!/usr/bin/env python3
"""
Build entity alias mapping for fuzzy name matching in text extraction.

This module creates a comprehensive mapping of entity names, labels, and aliases
to enable matching entity mentions in Wikipedia text even when names differ.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
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


def build_entity_aliases(jsonl_path: Path) -> Dict:
    """Build comprehensive entity alias mapping from JSONL data.

    Reads the wiki_entities_with_links.jsonl file and extracts all possible
    name variants for each entity (title, labels, aliases) to enable fuzzy
    matching during relationship extraction.

    Args:
        jsonl_path: Path to wiki_entities_with_links.jsonl file.

    Returns:
        Dictionary with structure:
        {
            "entity_map": {
                entity_id: {
                    "title": str,
                    "labels": Dict[str, str],  # lang -> label
                    "aliases": Dict[str, List[str]],  # lang -> [alias1, ...]
                    "category": str
                }
            },
            "name_to_entities": {
                normalized_name: [entity_id1, entity_id2, ...]
            }
        }

    Raises:
        FileNotFoundError: If jsonl_path does not exist.
    """
    entity_map: Dict[int, Dict] = {}
    name_to_entities: Dict[str, List[int]] = {}
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")
    
    print(f"Reading entities from {jsonl_path}...")
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing entities", unit="entity"):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract entity ID
            wp = obj.get("wikipedia", {})
            if not isinstance(wp, dict):
                continue
            
            entity_id = wp.get("id")
            if not isinstance(entity_id, int):
                continue
            
            # Extract title
            title = wp.get("title", "")
            if not isinstance(title, str):
                title = ""
            
            # Extract Wikidata labels and aliases
            wd = obj.get("wikidata", {})
            labels = {}
            aliases = {}
            
            if isinstance(wd, dict):
                # Labels: {"lang": {"value": "label"}} or {"lang": "label"}
                wd_labels = wd.get("labels", {})
                if isinstance(wd_labels, dict):
                    for lang, label_info in wd_labels.items():
                        if isinstance(label_info, dict):
                            label_val = label_info.get("value", "")
                        elif isinstance(label_info, str):
                            label_val = label_info
                        else:
                            continue
                        if label_val:
                            labels[lang] = label_val
                
                # Aliases: {"lang": [{"value": "alias"}, ...]} or {"lang": ["alias", ...]}
                wd_aliases = wd.get("aliases", {})
                if isinstance(wd_aliases, dict):
                    for lang, alias_list in wd_aliases.items():
                        if not isinstance(alias_list, list):
                            continue
                        alias_values = []
                        for alias_item in alias_list:
                            if isinstance(alias_item, dict):
                                alias_val = alias_item.get("value", "")
                            elif isinstance(alias_item, str):
                                alias_val = alias_item
                            else:
                                continue
                            if alias_val:
                                alias_values.append(alias_val)
                        if alias_values:
                            aliases[lang] = alias_values
            
            # Store entity information
            entity_map[entity_id] = {
                "title": title,
                "labels": labels,
                "aliases": aliases,
                "category": obj.get("category", "unknown")
            }
            
            # Build reverse mapping: normalized name -> entity IDs
            all_names: Set[str] = set()
            
            # Add title
            if title:
                all_names.add(normalize_name(title))
            
            # Add labels (prefer Vietnamese and English)
            for lang in ["vi", "en"]:
                if lang in labels:
                    label = labels[lang]
                    if label:
                        all_names.add(normalize_name(label))
            
            # Add aliases (prefer Vietnamese and English)
            for lang in ["vi", "en"]:
                if lang in aliases:
                    for alias in aliases[lang]:
                        if alias:
                            all_names.add(normalize_name(alias))
            
            # Map each name variant to this entity
            for name in all_names:
                if name:  # Skip empty names
                    if name not in name_to_entities:
                        name_to_entities[name] = []
                    if entity_id not in name_to_entities[name]:
                        name_to_entities[name].append(entity_id)
    
    return {
        "entity_map": entity_map,
        "name_to_entities": name_to_entities
    }


def main():
    """Main function to build entity aliases."""
    # Try main directory first, then old_code
    project_root = Path(__file__).resolve().parents[2]
    data_raw = Path(__file__).resolve().parent / "data_raw"
    input_file = data_raw / "wiki_entities_with_links.jsonl"
    if not input_file.exists():
        input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"
    output_file = Path(__file__).resolve().parent / "data" / "entity_aliases_map.json"
    
    print("=" * 80)
    print("Building Entity Alias Mapping")
    print("=" * 80)
    
    try:
        result = build_entity_aliases(input_file)
        
        # Save to JSON
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\nSuccessfully built alias mapping:")
        print(f"  Entities processed: {len(result['entity_map']):,}")
        print(f"  Unique name variants: {len(result['name_to_entities']):,}")
        print(f"  Output saved to: {output_file}")
        
        # Print some statistics
        entities_with_aliases = sum(
            1 for e in result['entity_map'].values()
            if e.get('aliases')
        )
        print(f"  Entities with aliases: {entities_with_aliases:,}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error building alias mapping: {e}")
        raise


if __name__ == "__main__":
    main()

