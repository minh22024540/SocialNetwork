#!/usr/bin/env python3
"""
Wikidata Entity Class - Production Implementation
Handles all possible property types and values from Wikidata API responses.

Based on official Wikidata API documentation at:
https://www.wikidata.org/wiki/Wikidata:Data_model
"""

import requests
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class StatementRank(Enum):
    """Quality rank of a statement."""
    PREFERRED = "preferred"
    NORMAL = "normal"
    DEPRECATED = "deprecated"


class DataType(Enum):
    """Wikidata datatypes."""
    WIKIBASE_ITEM = "wikibase-item"
    WIKIBASE_PROPERTY = "wikibase-property"
    STRING = "string"
    TIME = "time"
    MONOLINGUAL_TEXT = "monolingualtext"
    QUANTITY = "quantity"
    GLOBE_COORDINATE = "globe-coordinate"
    EXTERNAL_ID = "external-id"
    COMMONS_MEDIA = "commonsMedia"
    URL = "url"
    MATHEMATICAL_EXPRESSION = "math"


@dataclass
class TimeValue:
    """Wikidata time value."""
    time: str
    timezone: int
    before: int
    after: int
    precision: int
    calendarmodel: str
    
    def to_datetime(self) -> Optional[datetime]:
        """Convert to Python datetime if possible."""
        try:
            return datetime.fromisoformat(self.time.replace('Z', '+00:00'))
        except:
            return None
    
    def __str__(self) -> str:
        return self.time


@dataclass
class QuantityValue:
    """Wikidata quantity value."""
    amount: str
    unit: str
    upper_bound: Optional[str] = None
    lower_bound: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.amount} {self.unit}"


@dataclass
class GlobeCoordinateValue:
    """Wikidata globe coordinate value."""
    latitude: float
    longitude: float
    precision: Optional[float] = None
    globe: Optional[str] = None
    
    def __str__(self) -> str:
        return f"({self.latitude}, {self.longitude})"


@dataclass
class MonolingualTextValue:
    """Wikidata monolingual text value."""
    text: str
    language: str
    
    def __str__(self) -> str:
        return self.text


@dataclass
class EntityIdValue:
    """Wikidata entity ID value."""
    id: str
    entity_type: str
    numeric_id: Optional[int] = None
    
    def __str__(self) -> str:
        return self.id


@dataclass
class Snak:
    """A single snak (property-value pair)."""
    snaktype: str  # 'value', 'novalue', 'somevalue'
    property: str  # Property ID (e.g., 'P31')
    hash: Optional[str] = None
    datavalue: Optional[Dict[str, Any]] = None
    datatype: Optional[str] = None
    
    def get_value(self) -> Optional[Any]:
        """Extract the actual value from the datavalue structure."""
        if not self.datavalue:
            return None
        
        value = self.datavalue.get('value')
        value_type = self.datavalue.get('type')
        
        if value_type == DataType.TIME.value:
            return TimeValue(**value)
        elif value_type == DataType.QUANTITY.value:
            return QuantityValue(**value)
        elif value_type == DataType.GLOBE_COORDINATE.value:
            return GlobeCoordinateValue(**value)
        elif value_type == DataType.MONOLINGUAL_TEXT.value:
            return MonolingualTextValue(**value)
        elif value_type in [DataType.WIKIBASE_ITEM.value, DataType.WIKIBASE_PROPERTY.value]:
            return EntityIdValue(**value)
        elif value_type == DataType.STRING.value:
            return value
        elif value_type == DataType.EXTERNAL_ID.value:
            return value
        elif value_type == DataType.COMMONS_MEDIA.value:
            return value
        elif value_type == DataType.URL.value:
            return value
        else:
            return value  # Return as-is for unknown types


@dataclass
class Reference:
    """Reference backing a claim."""
    hash: str
    snaks: Dict[str, List[Dict]]
    snaks_order: List[str]
    
    def __post_init__(self):
        """Parse snaks into Snak objects."""
        self.snaks_parsed = {}
        for prop_id, snak_list in self.snaks.items():
            self.snaks_parsed[prop_id] = [Snak(**snak) for snak in snak_list]


@dataclass
class Statement:
    """A Wikidata statement (claim)."""
    id: str
    mainsnak: Dict[str, Any]
    type: str
    rank: str
    qualifiers: Optional[Dict[str, List[Dict]]] = None
    qualifiers_order: Optional[List[str]] = field(default_factory=list)
    references: Optional[List[Dict]] = None
    
    def __post_init__(self):
        """Parse mainsnak, qualifiers and references."""
        # Parse mainsnak
        self.mainsnak_obj = Snak(**self.mainsnak)
        
        # Parse qualifiers
        if self.qualifiers:
            self.qualifiers_parsed = {}
            for prop_id, snak_list in self.qualifiers.items():
                self.qualifiers_parsed[prop_id] = [Snak(**snak) for snak in snak_list]
        else:
            self.qualifiers_parsed = {}
        
        # Parse references
        if self.references:
            parsed_refs = []
            for ref in self.references:
                reference = Reference(
                    hash=ref.get("hash", ""),
                    snaks=ref.get("snaks", {}),
                    snaks_order=ref.get("snaks-order", ref.get("snaks_order", []))
                )
                parsed_refs.append(reference)
            self.references_parsed = parsed_refs
        else:
            self.references_parsed = []
    
    def get_main_value(self) -> Any:
        """Get the main value of this statement."""
        return self.mainsnak_obj.get_value()
    
    def get_qualifier_values(self, property_id: str) -> List[Any]:
        """Get qualifier values for a specific property."""
        if not hasattr(self, 'qualifiers_parsed') or property_id not in self.qualifiers_parsed:
            return []
        return [snak.get_value() for snak in self.qualifiers_parsed[property_id]]


@dataclass
class Sitelink:
    """A sitelink (Wikipedia link)."""
    site: str
    title: str
    badges: List[str] = field(default_factory=list)


@dataclass
class WikidataEntity:
    """
    Complete Wikidata entity representation.
    
    This class handles all possible property types and value structures
    returned by the Wikidata API.
    """
    id: str
    type: str
    labels: Dict[str, str] = field(default_factory=dict)
    descriptions: Dict[str, str] = field(default_factory=dict)
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    claims: Dict[str, List[Statement]] = field(default_factory=dict)
    statements: Dict[str, List[str]] = field(default_factory=dict)  # Simplified: property -> values
    sitelinks: Dict[str, Sitelink] = field(default_factory=dict)
    lastrevid: Optional[int] = None
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'WikidataEntity':
        """
        Create entity from Wikidata API JSON response.
        
        Args:
            json_data: JSON dict from API
            
        Returns:
            WikidataEntity instance
        """
        if "entities" not in json_data:
            raise ValueError("Invalid JSON structure: missing 'entities'")
        
        entity_id = list(json_data["entities"].keys())[0]
        entity_data = json_data["entities"][entity_id]
        
        # Parse labels
        labels = {}
        if "labels" in entity_data:
            if isinstance(entity_data["labels"], dict):
                # Standard format: {"lang": {"value": "label"}}
                for lang, label_info in entity_data["labels"].items():
                    labels[lang] = label_info.get("value", "")
            elif isinstance(entity_data["labels"], list):
                # Alternative format: [] (empty list)
                # In this case, labels is empty, so we leave labels as empty dict
                pass
        
        # Parse descriptions
        descriptions = {}
        if "descriptions" in entity_data:
            if isinstance(entity_data["descriptions"], dict):
                # Standard format: {"lang": {"value": "description"}}
                for lang, desc_info in entity_data["descriptions"].items():
                    descriptions[lang] = desc_info.get("value", "")
            elif isinstance(entity_data["descriptions"], list):
                # Alternative format: [] (empty list)
                # In this case, descriptions is empty, so we leave descriptions as empty dict
                pass
        
        # Parse aliases
        aliases = {}
        if "aliases" in entity_data:
            if isinstance(entity_data["aliases"], dict):
                # Standard format: {"lang": [{"value": "alias1"}, {"value": "alias2"}]}
                for lang, alias_list in entity_data["aliases"].items():
                    aliases[lang] = [alias.get("value", "") for alias in alias_list]
            elif isinstance(entity_data["aliases"], list):
                # Alternative format: [] (empty list)
                # In this case, aliases is empty, so we leave aliases as empty dict
                pass
        
        # Parse claims/statements
        claims = {}
        statements = {}  # Simplified statements: property -> list of values
        if "claims" in entity_data:
            for prop_id, claim_list in entity_data["claims"].items():
                parsed_claims = []
                simple_values = []
                
                for claim in claim_list:
                    # Handle hyphen in field names properly
                    statement = Statement(
                        id=claim.get("id", ""),
                        mainsnak=claim.get("mainsnak", {}),
                        type=claim.get("type", "statement"),
                        rank=claim.get("rank", "normal"),
                        qualifiers=claim.get("qualifiers"),
                        qualifiers_order=claim.get("qualifiers-order", claim.get("qualifiers_order", [])),
                        references=claim.get("references", [])
                    )
                    parsed_claims.append(statement)
                    
                    # Extract simple value from mainsnak
                    mainsnak = claim.get("mainsnak", {})
                    if mainsnak.get("snaktype") == "value" and "datavalue" in mainsnak:
                        value_data = mainsnak["datavalue"].get("value")
                        value_type = mainsnak["datavalue"].get("type")
                        
                        # Extract entity ID or text value based on type
                        if value_type in ["wikibase-item", "wikibase-property"] and isinstance(value_data, dict):
                            # For entity references (like Q10832752), extract the ID
                            entity_id = value_data.get("id")
                            if entity_id:
                                simple_values.append(entity_id)
                        elif value_type == "monolingualtext" and isinstance(value_data, dict):
                            # For text values (like native names), extract the text
                            text = value_data.get("text")
                            if text:
                                simple_values.append(text)
                        elif value_type == "string":
                            # For simple strings
                            if isinstance(value_data, str):
                                simple_values.append(value_data)
                        elif isinstance(value_data, (str, int, float)):
                            simple_values.append(str(value_data))
                
                claims[prop_id] = parsed_claims
                if simple_values:
                    statements[prop_id] = simple_values
        
        # Parse sitelinks
        sitelinks = {}
        if "sitelinks" in entity_data:
            for site, link_info in entity_data["sitelinks"].items():
                sitelinks[site] = Sitelink(
                    site=site,
                    title=link_info.get("title", ""),
                    badges=link_info.get("badges", [])
                )
        
        return cls(
            id=entity_data.get("id", entity_id),
            type=entity_data.get("type", "item"),
            labels=labels,
            descriptions=descriptions,
            aliases=aliases,
            claims=claims,
            statements=statements,
            sitelinks=sitelinks,
            lastrevid=entity_data.get("lastrevid")
        )
    
    def get_label(self, lang: str = "en") -> Optional[str]:
        """Get label in specified language."""
        return self.labels.get(lang)
    
    def get_description(self, lang: str = "en") -> Optional[str]:
        """Get description in specified language."""
        return self.descriptions.get(lang)
    
    def get_aliases(self, lang: str = "en") -> List[str]:
        """Get aliases in specified language."""
        return self.aliases.get(lang, [])
    
    def get_claims(self, property_id: str) -> List[Statement]:
        """Get all claims for a specific property."""
        return self.claims.get(property_id, [])
    
    def get_claim_values(self, property_id: str) -> List[Any]:
        """Get values of all claims for a specific property."""
        return [claim.get_main_value() for claim in self.get_claims(property_id)]
    
    def get_claim_value(self, property_id: str, index: int = 0) -> Optional[Any]:
        """Get a specific claim value by property ID and index."""
        claims = self.get_claims(property_id)
        if index < len(claims):
            return claims[index].get_main_value()
        return None
    
    def get_statement_values(self, property_id: str) -> List[str]:
        """Get all statement values for a specific property (simplified format)."""
        return self.statements.get(property_id, [])
    
    def get_statement_value(self, property_id: str, index: int = 0) -> Optional[str]:
        """Get a specific statement value by property ID and index."""
        statements = self.get_statement_values(property_id)
        if index < len(statements):
            return statements[index]
        return None
    
    def get_wikipedia_link(self, lang: str = "en") -> Optional[str]:
        """Get Wikipedia article title for specified language."""
        sitelink = self.sitelinks.get(f"{lang}wiki")
        return sitelink.title if sitelink else None
    
    def get_all_wikipedia_links(self) -> Dict[str, str]:
        """Get all Wikipedia article titles."""
        return {
            site.replace("wiki", ""): sitelink.title
            for site, sitelink in self.sitelinks.items()
            if "wiki" in site
        }
    
    def count_claims(self) -> int:
        """Count total number of claims."""
        return sum(len(statements) for statements in self.claims.values())
    
    def count_sitelinks(self) -> int:
        """Count total number of sitelinks."""
        return len(self.sitelinks)
    
    def has_property(self, property_id: str) -> bool:
        """Check if entity has claims for a specific property."""
        return property_id in self.claims and len(self.claims[property_id]) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "labels": self.labels,
            "descriptions": self.descriptions,
            "aliases": self.aliases,
            "statements": self.statements,
            "claims_count": self.count_claims(),
            "sitelink_count": self.count_sitelinks()
        }
    
    def to_json(self) -> str:
        """Convert entity to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @staticmethod
    def fetch(qid: str, languages: Optional[str] = None) -> 'WikidataEntity':
        """
        Fetch entity from Wikidata API.
        
        Args:
            qid: Wikidata entity ID (e.g., "Q9199")
            languages: Comma-separated language codes (e.g., "en,vi,zh")
        
        Returns:
            WikidataEntity instance
        """
        api_url = "https://www.wikidata.org/w/api.php"
        
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "format": "json"
        }
        
        if languages:
            params["languages"] = languages
        
        headers = {"User-Agent": "WikidataEntity/1.0 (Python)"}
        
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        
        json_data = response.json()
        return WikidataEntity.from_json(json_data)
    
    def __repr__(self) -> str:
        return f"WikidataEntity(id={self.id}, type={self.type}, claims={self.count_claims()}, statements={len(self.statements)}, sitelinks={self.count_sitelinks()})"


if __name__ == "__main__":
    # Example usage
    print("Testing Wikidata Entity Class")
    print("=" * 80)
    
    # Fetch an entity
    entity = WikidataEntity.fetch("Q9199", languages="en,vi,zh")
    
    print(f"\nEntity: {entity}")
    print(f"English name: {entity.get_label('en')}")
    print(f"Vietnamese name: {entity.get_label('vi')}")
    print(f"Chinese name: {entity.get_label('zh')}")
    print(f"Description: {entity.get_description('en')}")
    print(f"Total claims: {entity.count_claims()}")
    print(f"Total sitelinks: {entity.count_sitelinks()}")
    
    # Get specific claim values
    print(f"\nInstance of (P31): {entity.get_claim_value('P31')}")
    print(f"Country (P17): {entity.get_claim_values('P17')}")
    
    # Get simplified statement values
    print(f"\nInstance of (simplified) (P31): {entity.get_statement_values('P31')}")
    print(f"Country (simplified) (P17): {entity.get_statement_values('P17')}")
    
    # Get Wikipedia links
    print(f"\nEnglish Wikipedia: {entity.get_wikipedia_link('en')}")
    print(f"Vietnamese Wikipedia: {entity.get_wikipedia_link('vi')}")
    print(f"Chinese Wikipedia: {entity.get_wikipedia_link('zh')}")

