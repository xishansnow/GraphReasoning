"""
Entity Extraction Module
========================

Extracts named entities (persons, places, events, dates) from historical texts.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    """Types of entities in historical documents."""
    PERSON = "person"
    PLACE = "place"
    EVENT = "event"
    DATE = "date"
    ORGANIZATION = "organization"
    DOCUMENT = "document"
    CONCEPT = "concept"
    ARTIFACT = "artifact"


@dataclass
class Entity:
    """Represents an extracted entity."""
    
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    context: Optional[str] = None
    source_document: Optional[str] = None
    start_position: int = -1
    end_position: int = -1
    confidence: float = 1.0
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "description": self.description,
            "context": self.context,
            "source_document": self.source_document,
            "confidence": self.confidence,
            "attributes": self.attributes
        }


class EntityExtractor:
    """Extracts entities from historical texts."""
    
    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None):
        """
        Initialize entity extractor.
        
        Args:
            use_llm: Whether to use LLM for advanced extraction
            llm_provider: LLM provider ('openai', 'huggingface', etc.)
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.extracted_entities: Dict[str, Entity] = {}
    
    def extract_entities(self, text: str, document_source: Optional[str] = None,
                        entity_types: Optional[List[EntityType]] = None) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            document_source: Source document identifier
            entity_types: Types of entities to extract (None = all)
        
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Use basic extraction if no LLM
        if not self.use_llm:
            entities = self._extract_entities_regex(text, document_source)
        else:
            entities = self._extract_entities_llm(text, document_source)
        
        # Filter by type if specified
        if entity_types:
            entities = [e for e in entities if e.entity_type in entity_types]
        
        return entities
    
    def _extract_entities_regex(self, text: str, 
                               document_source: Optional[str] = None) -> List[Entity]:
        """
        Extract entities using regex patterns (basic approach).
        
        Args:
            text: Input text
            document_source: Source document identifier
        
        Returns:
            List of entities
        """
        import re
        from datetime import datetime
        
        entities = []
        
        # Extract dates (simple pattern: YYYY or month/year patterns)
        date_pattern = r'\b(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+)?(?:\d{1,2},?\s+)?(?:1[0-9]{3}|20\d{2})\b'
        for match in re.finditer(date_pattern, text):
            entities.append(Entity(
                name=match.group(),
                entity_type=EntityType.DATE,
                context=self._get_context(text, match.start(), match.end()),
                source_document=document_source,
                start_position=match.start(),
                end_position=match.end(),
                confidence=0.7
            ))
        
        # Extract capitalized phrases (potential names/places)
        # Pattern: 1-3 capitalized words
        name_pattern = r'\b(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+\b'
        seen_names = set()
        
        for match in re.finditer(name_pattern, text):
            name = match.group().strip()
            
            # Skip common words
            if name in {'The', 'And', 'Or', 'In', 'Is', 'Are', 'This', 'That'}:
                continue
            
            if name not in seen_names:
                seen_names.add(name)
                # Heuristic: Names with titles or all caps likely persons
                if any(title in name for title in ['King', 'Queen', 'Emperor', 'Saint', 'General', 'Dr.']):
                    entity_type = EntityType.PERSON
                elif any(place in name for place in ['River', 'Mountain', 'City', 'Valley', 'Lake']):
                    entity_type = EntityType.PLACE
                else:
                    # Default to person for capitalized names
                    entity_type = EntityType.PERSON if len(name.split()) <= 3 else EntityType.PLACE
                
                entities.append(Entity(
                    name=name,
                    entity_type=entity_type,
                    context=self._get_context(text, match.start(), match.end()),
                    source_document=document_source,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.6
                ))
        
        return entities
    
    def _extract_entities_llm(self, text: str, 
                             document_source: Optional[str] = None) -> List[Entity]:
        """
        Extract entities using LLM (advanced approach).
        
        Args:
            text: Input text
            document_source: Source document identifier
        
        Returns:
            List of entities
        """
        # This would integrate with GraphReasoning.llm_providers
        # For now, return empty list as placeholder
        print("LLM-based extraction not yet implemented")
        return []
    
    def _get_context(self, text: str, start: int, end: int, 
                    context_window: int = 50) -> str:
        """
        Get context around an entity.
        
        Args:
            text: Full text
            start: Start position
            end: End position
            context_window: Characters of context on each side
        
        Returns:
            Context string
        """
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        
        return text[context_start:context_end].strip()
    
    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate or similar entities.
        
        Args:
            entities: List of entities
        
        Returns:
            Deduplicated list
        """
        seen = {}
        deduplicated = []
        
        for entity in entities:
            # Create a key based on name and type
            key = (entity.name.lower().strip(), entity.entity_type)
            
            if key not in seen:
                seen[key] = True
                deduplicated.append(entity)
            else:
                # If duplicate has higher confidence, replace
                existing_idx = next(i for i, e in enumerate(deduplicated) 
                                   if e.name.lower() == entity.name.lower() 
                                   and e.entity_type == entity.entity_type)
                if entity.confidence > deduplicated[existing_idx].confidence:
                    deduplicated[existing_idx] = entity
        
        return deduplicated
    
    def filter_entities_by_confidence(self, entities: List[Entity], 
                                     min_confidence: float = 0.5) -> List[Entity]:
        """
        Filter entities by confidence threshold.
        
        Args:
            entities: List of entities
            min_confidence: Minimum confidence score
        
        Returns:
            Filtered list
        """
        return [e for e in entities if e.confidence >= min_confidence]
