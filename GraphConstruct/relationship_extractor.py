"""
Relationship Extraction Module
==============================

Extracts relationships between entities from historical texts.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .entity_extractor import Entity, EntityType


class RelationType(Enum):
    """Types of relationships between entities."""
    # Temporal relationships
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    CONTEMPORARY_WITH = "contemporary_with"
    
    # Hierarchical relationships
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    ANCESTOR_OF = "ancestor_of"
    
    # Social relationships
    MARRIED_TO = "married_to"
    SIBLING_OF = "sibling_of"
    COLLEAGUE_OF = "colleague_of"
    RIVAL_OF = "rival_of"
    ALLY_OF = "ally_of"
    ENEMY_OF = "enemy_of"
    
    # Organizational relationships
    MEMBER_OF = "member_of"
    LEADER_OF = "leader_of"
    FOUNDED = "founded"
    PART_OF = "part_of"
    
    # Locational relationships
    LOCATED_IN = "located_in"
    BORDERS = "borders"
    NEAR = "near"
    CONTAINS = "contains"
    
    # Event relationships
    PARTICIPATED_IN = "participated_in"
    CAUSED = "caused"
    RESULTED_IN = "resulted_in"
    OCCURRED_IN = "occurred_in"
    
    # General relationships
    RELATED_TO = "related_to"
    INFLUENCED = "influenced"
    OPPOSED = "opposed"
    SUPPORTED = "supported"


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    
    subject: Entity
    predicate: RelationType
    object: Entity
    context: Optional[str] = None
    confidence: float = 1.0
    source_document: Optional[str] = None
    supporting_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "subject": self.subject.name,
            "subject_type": self.subject.entity_type.value,
            "predicate": self.predicate.value,
            "object": self.object.name,
            "object_type": self.object.entity_type.value,
            "confidence": self.confidence,
            "context": self.context
        }
    
    def to_triplet(self) -> Tuple[str, str, str]:
        """Convert to RDF triplet format."""
        return (
            self.subject.name,
            self.predicate.value,
            self.object.name
        )


class RelationshipExtractor:
    """Extracts relationships between entities from texts."""
    
    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None):
        """
        Initialize relationship extractor.
        
        Args:
            use_llm: Whether to use LLM for advanced extraction
            llm_provider: LLM provider ('openai', 'huggingface', etc.)
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        # Relationship patterns for regex-based extraction
        self.relationship_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[RelationType, List[str]]:
        """Initialize regex patterns for relationship detection."""
        return {
            RelationType.MARRIED_TO: [
                r'married\s+to', r'wife\s+of', r'husband\s+of', 
                r'spouse\s+of', r'marriage\s+to', r'wed\s+to'
            ],
            RelationType.PARENT_OF: [
                r'(?:father|mother|parent)\s+of', r'(?:son|daughter|child)\s+of',
                r'begat', r'fathered', r'mothered'
            ],
            RelationType.LOCATED_IN: [
                r'(?:located|situated|based)\s+in', r'in\s+(?:the\s+)?city\s+of',
                r'capital\s+of', r'from\s+(?:the\s+)?', r'lived\s+in', r'born\s+in'
            ],
            RelationType.MEMBER_OF: [
                r'member\s+of', r'joined', r'part\s+of', r'belong[s]?\s+to',
                r'(?:was\s+)?elected\s+to'
            ],
            RelationType.LEADER_OF: [
                r'(?:king|queen|emperor|president|governor)\s+of',
                r'led', r'ruled', r'commanded', r'headed'
            ],
            RelationType.PARTICIPATED_IN: [
                r'participated\s+in', r'fought\s+in', r'attended',
                r'took\s+part\s+in', r'engaged\s+in'
            ],
            RelationType.CAUSED: [
                r'caused', r'led\s+to', r'resulted\s+in', r'triggered',
                r'provoked', r'sparked'
            ],
        }
    
    def extract_relationships(self, text: str, entities: List[Entity],
                            document_source: Optional[str] = None) -> List[Relationship]:
        """
        Extract relationships between entities from text.
        
        Args:
            text: Input text
            entities: List of entities already extracted
            document_source: Source document identifier
        
        Returns:
            List of relationships
        """
        if not self.use_llm:
            return self._extract_relationships_pattern(text, entities, document_source)
        else:
            return self._extract_relationships_llm(text, entities, document_source)
    
    def _extract_relationships_pattern(self, text: str, entities: List[Entity],
                                      document_source: Optional[str] = None) -> List[Relationship]:
        """
        Extract relationships using pattern matching.
        
        Args:
            text: Input text
            entities: List of entities
            document_source: Source document identifier
        
        Returns:
            List of relationships
        """
        import re
        
        relationships = []
        entity_map = {e.name: e for e in entities}
        
        # For each relationship type and its patterns
        for rel_type, patterns in self.relationship_patterns.items():
            pattern_str = "|".join(f"({p})" for p in patterns)
            
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                # Find nearby entities
                match_pos = match.start()
                nearby_entities = self._find_nearby_entities(
                    entities, text, match_pos, window=200
                )
                
                if len(nearby_entities) >= 2:
                    subject, obj = nearby_entities[0], nearby_entities[1]
                    
                    relationship = Relationship(
                        subject=subject,
                        predicate=rel_type,
                        object=obj,
                        context=self._get_context(text, match.start(), match.end()),
                        supporting_text=match.group(),
                        source_document=document_source,
                        confidence=0.7
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_relationships_llm(self, text: str, entities: List[Entity],
                                  document_source: Optional[str] = None) -> List[Relationship]:
        """
        Extract relationships using LLM.
        
        Args:
            text: Input text
            entities: List of entities
            document_source: Source document identifier
        
        Returns:
            List of relationships
        """
        # This would integrate with GraphReasoning.llm_providers
        print("LLM-based relationship extraction not yet implemented")
        return []
    
    def _find_nearby_entities(self, entities: List[Entity], text: str,
                             position: int, window: int = 100) -> List[Entity]:
        """
        Find entities near a specific position in text.
        
        Args:
            entities: List of entities
            text: Full text
            position: Position to search around
            window: Search window size
        
        Returns:
            List of nearby entities sorted by distance
        """
        nearby = []
        
        for entity in entities:
            if entity.start_position >= 0:
                distance = abs(entity.start_position - position)
                if distance < window:
                    nearby.append((distance, entity))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[0])
        return [e for _, e in nearby]
    
    def _get_context(self, text: str, start: int, end: int,
                    context_window: int = 50) -> str:
        """Get context around a relationship mention."""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end].strip()
    
    def deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Remove duplicate relationships.
        
        Args:
            relationships: List of relationships
        
        Returns:
            Deduplicated list
        """
        seen = {}
        deduplicated = []
        
        for rel in relationships:
            # Create key: (subject, predicate, object)
            key = (rel.subject.name, rel.predicate, rel.object.name)
            
            if key not in seen:
                seen[key] = True
                deduplicated.append(rel)
            else:
                # If duplicate has higher confidence, replace
                existing_idx = next(
                    i for i, r in enumerate(deduplicated)
                    if (r.subject.name == rel.subject.name and
                        r.predicate == rel.predicate and
                        r.object.name == rel.object.name)
                )
                if rel.confidence > deduplicated[existing_idx].confidence:
                    deduplicated[existing_idx] = rel
        
        return deduplicated
    
    def filter_relationships_by_confidence(self, relationships: List[Relationship],
                                          min_confidence: float = 0.5) -> List[Relationship]:
        """
        Filter relationships by confidence threshold.
        
        Args:
            relationships: List of relationships
            min_confidence: Minimum confidence score
        
        Returns:
            Filtered list
        """
        return [r for r in relationships if r.confidence >= min_confidence]
