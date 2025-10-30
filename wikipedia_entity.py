#!/usr/bin/env python3
"""
Wikipedia Entity Class
Represents a Wikipedia article from the dump file.
"""

from dataclasses import dataclass


@dataclass
class WikipediaEntity:
    """
    Wikipedia article data from dump file.
    
    Attributes:
        id: Wikipedia page ID
        namespace: Article namespace (0 for main articles)
        title: Article title
        full_text: Complete article content
    """
    id: int
    namespace: int
    title: str
    full_text: str
    
    def is_main_article(self) -> bool:
        """Check if this is a main namespace article (not Talk, Template, etc.)."""
        return self.namespace == 0
    
    def __repr__(self) -> str:
        return f"WikipediaEntity(id={self.id}, title='{self.title}')"

