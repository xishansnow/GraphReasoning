"""
Ontology/Schema Generation Module
==================================

Automatically generate ontology/schema from:
1. Top-down: Capability questions or requirements
2. Bottom-up: Analysis of extracted triples

This module provides utilities for schema discovery and generation.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
import json
from collections import Counter, defaultdict
import re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from Llms.llm_providers import get_generate_fn


class EntityTypeInferenceMethod(Enum):
    """Methods for inferring entity types from triples."""
    FREQUENCY = "frequency"  # Based on occurrence frequency
    PATTERN = "pattern"      # Based on naming patterns
    LLM = "llm"             # Using LLM to infer types
    HYBRID = "hybrid"       # Combination of above


@dataclass
class InferredEntityType:
    """Represents an inferred entity type."""
    name: str
    frequency: int
    examples: List[str]
    description: str = ""
    properties: List[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = []


@dataclass
class InferredRelationType:
    """Represents an inferred relation type."""
    name: str
    domain: str
    range: str
    frequency: int
    examples: List[Tuple[str, str]] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class TripleAnalyzer:
    """Analyze triples to extract entity and relation patterns."""
    
    def __init__(self):
        self.entity_counter = Counter()
        self.relation_counter = Counter()
        self.entity_type_patterns = defaultdict(list)
        self.relation_domain_range = defaultdict(lambda: defaultdict(int))
    
    def analyze_triples(self, triples: List[Dict]) -> Dict:
        """
        Analyze triples to extract patterns.
        
        Args:
            triples (List[Dict]): List of triple dictionaries with keys:
                ["node_1", "node_2", "edge", "node_1_type", "node_2_type"]
        
        Returns:
            Dict: Analysis results including entity and relation statistics
        """
        results = {
            "total_triples": len(triples),
            "unique_entities": set(),
            "unique_relations": set(),
            "entity_distribution": {},
            "relation_distribution": {},
            "domain_range_pairs": defaultdict(lambda: defaultdict(int)),
        }
        
        for triple in triples:
            node_1 = triple.get("node_1", "")
            node_2 = triple.get("node_2", "")
            edge = triple.get("edge", "")
            node_1_type = triple.get("node_1_type", "Unknown")
            node_2_type = triple.get("node_2_type", "Unknown")
            
            # Track entities
            results["unique_entities"].add(node_1)
            results["unique_entities"].add(node_2)
            self.entity_counter[node_1] += 1
            self.entity_counter[node_2] += 1
            
            # Track relations
            results["unique_relations"].add(edge)
            self.relation_counter[edge] += 1
            
            # Track domain-range pairs
            results["domain_range_pairs"][edge][(node_1_type, node_2_type)] += 1
            self.relation_domain_range[edge][(node_1_type, node_2_type)] += 1
        
        results["entity_distribution"] = dict(self.entity_counter)
        results["relation_distribution"] = dict(self.relation_counter)
        results["unique_entities_count"] = len(results["unique_entities"])
        results["unique_relations_count"] = len(results["unique_relations"])
        
        return results
    
    def infer_entity_types(self, triples: List[Dict], 
                          method: EntityTypeInferenceMethod = EntityTypeInferenceMethod.PATTERN,
                          min_frequency: int = 2,
                          verbatim: bool = False) -> Dict[str, InferredEntityType]:
        """
        Infer entity types from triples.
        
        Args:
            triples (List[Dict]): List of triples
            method (EntityTypeInferenceMethod): Inference method to use
            min_frequency (int): Minimum entity frequency to include
            verbatim (bool): Print verbose output
        
        Returns:
            Dict[str, InferredEntityType]: Inferred entity types
        """
        entity_types = {}
        entity_examples = defaultdict(set)
        
        # Collect entities and their types
        for triple in triples:
            node_1 = triple.get("node_1", "")
            node_2 = triple.get("node_2", "")
            node_1_type = triple.get("node_1_type", "")
            node_2_type = triple.get("node_2_type", "")
            
            if node_1_type:
                entity_examples[node_1_type].add(node_1)
            if node_2_type:
                entity_examples[node_2_type].add(node_2)
        
        # Create entity types based on method
        for entity_type, examples in entity_examples.items():
            examples_list = list(examples)[:5]  # Top 5 examples
            
            inferred = InferredEntityType(
                name=entity_type,
                frequency=len(examples),
                examples=examples_list,
                confidence=min(len(examples) / max(1, len(triples)), 1.0)
            )
            
            entity_types[entity_type] = inferred
            
            if verbatim:
                print(f"✅ Entity Type: {entity_type}")
                print(f"   Frequency: {inferred.frequency}, Examples: {examples_list}")
        
        return entity_types
    
    def infer_relation_types(self, triples: List[Dict],
                            min_frequency: int = 2,
                            verbatim: bool = False) -> Dict[str, InferredRelationType]:
        """
        Infer relation types from triples.
        
        Args:
            triples (List[Dict]): List of triples
            min_frequency (int): Minimum relation frequency to include
            verbatim (bool): Print verbose output
        
        Returns:
            Dict[str, InferredRelationType]: Inferred relation types
        """
        relation_types = {}
        relation_patterns = defaultdict(lambda: defaultdict(int))
        relation_examples = defaultdict(list)
        
        for triple in triples:
            node_1 = triple.get("node_1", "")
            node_2 = triple.get("node_2", "")
            edge = triple.get("edge", "")
            node_1_type = triple.get("node_1_type", "Unknown")
            node_2_type = triple.get("node_2_type", "Unknown")
            
            # Track domain-range patterns
            relation_patterns[edge][(node_1_type, node_2_type)] += 1
            
            # Track examples
            if len(relation_examples[edge]) < 3:
                relation_examples[edge].append((node_1, node_2))
        
        # Create relation types
        for edge, patterns in relation_patterns.items():
            # Get most common domain-range pair
            most_common_pair = max(patterns.items(), key=lambda x: x[1])
            domain, range_ = most_common_pair[0]
            frequency = sum(patterns.values())
            
            if frequency >= min_frequency:
                inferred = InferredRelationType(
                    name=edge,
                    domain=domain,
                    range=range_,
                    frequency=frequency,
                    examples=relation_examples[edge],
                    confidence=most_common_pair[1] / frequency
                )
                
                relation_types[edge] = inferred
                
                if verbatim:
                    print(f"✅ Relation Type: {edge}")
                    print(f"   Domain: {domain}, Range: {range_}")
                    print(f"   Frequency: {frequency}, Confidence: {inferred.confidence:.2f}")
        
        return relation_types

####################################################################
# Top-down Advanced CQ-based Ontology Generation Methods
####################################################################

class TopDownOntologyExtractor:
    """
    Extract ontology from capability questions (top-down approach).
    
    Supports multiple generation methods:
    - 'pattern': Pattern matching (default, fast, rule-based)
    - 'cqbycq': Iterative processing with memory (LLM-based)
    - 'memoryless': Independent processing with merging (LLM-based)
    - 'ontogenia': All-at-once processing (LLM-based)
    """
    
    def __init__(self, generate_fn=None, method: str = 'pattern'):
        """
        Initialize extractor.
        
        Args:
            generate_fn: LLM generation function (required for LLM-based methods)
            method: Extraction method - 'pattern', 'cqbycq', 'memoryless', or 'ontogenia'
        """
        self.generate_fn = generate_fn
        self.method = method
        
        # Initialize LLM-based generators if needed
        if method in ['cqbycq', 'memoryless', 'ontogenia']:
            if generate_fn is None:
                generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
                self.generate_fn = generate_fn
            
            if method == 'cqbycq':
                self._llm_generator = CQbyCQGenerator(generate_fn=generate_fn)
            elif method == 'memoryless':
                self._llm_generator = MemorylessCQbyCQGenerator(generate_fn=generate_fn)
            elif method == 'ontogenia':
                self._llm_generator = OntogeniaGenerator(generate_fn=generate_fn)
        else:
            self._llm_generator = None
    
    def extract_from_competency_questions(self, 
                                         questions: List[str],
                                         verbatim: bool = False,
                                         method: str = None) -> Dict:
        """
        Extract entity and relation types from competency questions.
        
        Args:
            questions (List[str]): List of competency questions
                Example: [
                    "Which persons work for which organizations?",
                    "What are the events participated by persons?",
                    "Which organizations are located in which locations?"
                ]
            verbatim (bool): Print verbose output
            method (str): Override default method - 'pattern', 'cqbycq', 'memoryless', or 'ontogenia'
        
        Returns:
            Dict: Extracted ontology including entity types and relations
        """
        # Use override method if provided
        extraction_method = method if method is not None else self.method
        
        if verbatim:
            print(f"🔬 Using extraction method: {extraction_method.upper()}")
        
        # Route to appropriate extraction method
        if extraction_method == 'pattern':
            return self._extract_pattern_based(questions, verbatim)
        elif extraction_method == 'cqbycq':
            return self._extract_cqbycq(questions, verbatim)
        elif extraction_method == 'memoryless':
            return self._extract_memoryless(questions, verbatim)
        elif extraction_method == 'ontogenia':
            return self._extract_ontogenia(questions, verbatim)
        else:
            raise ValueError(f"Unknown method: {extraction_method}. "
                           f"Use 'pattern', 'cqbycq', 'memoryless', or 'ontogenia'")
    
    def _extract_pattern_based(self, questions: List[str], verbatim: bool = False) -> Dict:
        """Extract using pattern matching (original method)."""
        ontology = {
            "entity_types": {},
            "relation_types": {},
            "competency_questions": questions,
            "metadata": {
                "method": "pattern",
                "description": "Pattern-based extraction (rule-based)"
            }
        }
        
        # Extract entities
        entities = self._extract_entities_from_questions(questions, verbatim)
        ontology["entity_types"] = entities
        
        # Extract relations
        relations = self._extract_relations_from_questions(questions, entities, verbatim)
        ontology["relation_types"] = relations
        
        return ontology
    
    def _extract_cqbycq(self, questions: List[str], verbatim: bool = False) -> Dict:
        """Extract using CQbyCQ method."""
        if self._llm_generator is None or not isinstance(self._llm_generator, CQbyCQGenerator):
            if self.generate_fn is None:
                self.generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
            self._llm_generator = CQbyCQGenerator(generate_fn=self.generate_fn, verbatim=verbatim)
        
        return self._llm_generator.generate_ontology(questions)
    
    def _extract_memoryless(self, questions: List[str], verbatim: bool = False) -> Dict:
        """Extract using Memoryless CQbyCQ method."""
        if self._llm_generator is None or not isinstance(self._llm_generator, MemorylessCQbyCQGenerator):
            if self.generate_fn is None:
                self.generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
            self._llm_generator = MemorylessCQbyCQGenerator(generate_fn=self.generate_fn, verbatim=verbatim)
        
        return self._llm_generator.generate_ontology(questions)
    
    def _extract_ontogenia(self, questions: List[str], verbatim: bool = False) -> Dict:
        """Extract using Ontogenia method."""
        if self._llm_generator is None or not isinstance(self._llm_generator, OntogeniaGenerator):
            if self.generate_fn is None:
                self.generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
            self._llm_generator = OntogeniaGenerator(generate_fn=self.generate_fn, verbatim=verbatim)
        
        return self._llm_generator.generate_ontology(questions)
    
    def compare_methods(self, questions: List[str], verbatim: bool = False) -> Dict:
        """
        Compare all extraction methods on the same questions.
        
        Args:
            questions: List of competency questions
            verbatim: Print detailed logs
        
        Returns:
            Dict: Comparison results with ontologies from all methods
        """
        if verbatim:
            print("="*70)
            print("COMPARING ALL EXTRACTION METHODS")
            print("="*70)
        
        results = {
            "pattern": None,
            "cqbycq": None,
            "memoryless": None,
            "ontogenia": None,
            "comparison_stats": {}
        }
        
        # Pattern-based
        if verbatim:
            print("\n" + "="*70)
            print("METHOD 1: Pattern-based (Rule-based)")
            print("="*70)
        results["pattern"] = self._extract_pattern_based(questions, verbatim)
        
        # CQbyCQ
        if verbatim:
            print("\n" + "="*70)
            print("METHOD 2: CQbyCQ (Iterative with Memory)")
            print("="*70)
        results["cqbycq"] = self._extract_cqbycq(questions, verbatim)
        
        # Memoryless
        if verbatim:
            print("\n" + "="*70)
            print("METHOD 3: Memoryless CQbyCQ (Independent Processing)")
            print("="*70)
        results["memoryless"] = self._extract_memoryless(questions, verbatim)
        
        # Ontogenia
        if verbatim:
            print("\n" + "="*70)
            print("METHOD 4: Ontogenia (All-at-Once)")
            print("="*70)
        results["ontogenia"] = self._extract_ontogenia(questions, verbatim)
        
        # Compute comparison stats
        for method_name in ["pattern", "cqbycq", "memoryless", "ontogenia"]:
            onto = results[method_name]
            results["comparison_stats"][f"{method_name}_entities"] = len(onto.get("entity_types", {}))
            results["comparison_stats"][f"{method_name}_relations"] = len(onto.get("relation_types", {}))
        
        # Summary
        if verbatim:
            print("\n" + "="*70)
            print("COMPARISON SUMMARY")
            print("="*70)
            
            for method in ["pattern", "cqbycq", "memoryless", "ontogenia"]:
                onto = results[method]
                print(f"\n{method.upper()}:")
                print(f"  Entity types: {len(onto.get('entity_types', {}))}")
                print(f"  Relation types: {len(onto.get('relation_types', {}))}")
                if onto.get('entity_types'):
                    entities = list(onto['entity_types'].keys())[:5]
                    print(f"  Sample entities: {', '.join(entities)}")
        
        return results
    
    def _extract_entities_from_questions(self, questions: List[str], 
                                        verbatim: bool = False) -> Dict:
        """Extract entity types from questions using pattern matching."""
        entities = {}
        entity_pattern = r'\b(Person|Organization|Location|Event|Concept|Document|Project|Date|Time|Thing|Resource)\s*s?\b'
        
        for question in questions:
            matches = re.findall(entity_pattern, question, re.IGNORECASE)
            for match in matches:
                entity_type = match.capitalize()
                if entity_type not in entities:
                    entities[entity_type] = {
                        "name": entity_type,
                        "properties": self._infer_properties_for_entity(entity_type),
                        "description": f"Extracted from competency question"
                    }
        
        if verbatim:
            print(f"✅ Extracted {len(entities)} entity types from questions:")
            for ent in entities:
                print(f"   - {ent}: {entities[ent]['properties']}")
        
        return entities
    
    def _extract_relations_from_questions(self, questions: List[str],
                                         entities: Dict,
                                         verbatim: bool = False) -> Dict:
        """Extract relation types from questions using pattern matching."""
        relations = {}
        relation_pattern = r'(work[s]?\s+for|located\s+in|participate[sd]?\s+in|part\s+of|related\s+to|knows|manages|contains|creates)'
        
        for question in questions:
            matches = re.findall(relation_pattern, question, re.IGNORECASE)
            for match in matches:
                relation_name = match.lower().replace(" ", "_")
                
                if relation_name not in relations:
                    # Try to infer domain and range
                    domain, range_ = self._infer_domain_range(question, relation_name, entities)
                    
                    relations[relation_name] = {
                        "name": relation_name,
                        "domain": domain,
                        "range": range_,
                        "description": f"Extracted from: {question}"
                    }
        
        if verbatim:
            print(f"\n✅ Extracted {len(relations)} relation types from questions:")
            for rel, info in relations.items():
                print(f"   - {rel}: {info['domain']} → {info['range']}")
        
        return relations
    
    def _infer_properties_for_entity(self, entity_type: str) -> List[str]:
        """Infer common properties for entity type."""
        default_properties = {
            "Person": ["name", "age", "role", "email"],
            "Organization": ["name", "type", "founded_date", "location"],
            "Location": ["name", "type", "coordinates"],
            "Event": ["name", "date", "location", "description"],
            "Concept": ["name", "definition"],
            "Document": ["title", "content", "author", "date"],
            "Project": ["name", "description", "start_date", "status"],
        }
        return default_properties.get(entity_type, ["name", "description"])
    
    def _infer_domain_range(self, question: str, relation: str, 
                           entities: Dict) -> Tuple[str, str]:
        """Infer domain and range for a relation from context."""
        # Simple heuristic: look for entity types before and after the relation
        entity_names = list(entities.keys())
        
        # Default mappings
        relation_mappings = {
            "work_for": ("Person", "Organization"),
            "works_for": ("Person", "Organization"),
            "located_in": ("Organization", "Location"),
            "participate_in": ("Person", "Event"),
            "participated_in": ("Person", "Event"),
            "part_of": ("Organization", "Organization"),
            "related_to": ("Concept", "Concept"),
        }
        
        return relation_mappings.get(relation, ("Thing", "Thing"))




class CQbyCQGenerator:
    """
    CQbyCQ (Competency Question by Competency Question) Method
    
    Iteratively refines ontology by processing one competency question at a time,
    accumulating knowledge and context from previous questions.
    
    Based on: "Ontology Generation using Large Language Models"
    
    Algorithm:
    1. Start with empty ontology
    2. For each CQ:
       - Present current ontology + new CQ to LLM
       - LLM proposes updates (new classes, properties, relations)
       - Integrate updates into ontology
    3. Return final comprehensive ontology
    
    Advantages:
    - Builds rich context from previous questions
    - Allows refinement and evolution
    - Better semantic consistency
    
    Disadvantages:
    - Can accumulate errors
    - May become biased by early decisions
    - Slower due to iterative nature
    """
    
    def __init__(self, generate_fn=None, verbatim: bool = False):
        """
        Initialize CQbyCQ generator.
        
        Args:
            generate_fn: LLM generation function
            verbatim: Print detailed logs
        """
        if generate_fn is None:
            generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
        self.generate_fn = generate_fn
        self.verbatim = verbatim
    
    def generate_ontology(self, competency_questions: List[str]) -> Dict:
        """
        Generate ontology using CQbyCQ method.
        
        Args:
            competency_questions: List of competency questions
        
        Returns:
            Dict: Generated ontology with entity_types and relation_types
        """
        # Initialize empty ontology
        ontology = {
            "entity_types": {},
            "relation_types": {},
            "metadata": {
                "method": "CQbyCQ",
                "total_cqs": len(competency_questions),
                "processed_cqs": []
            }
        }
        
        if self.verbatim:
            print(f"🔬 CQbyCQ Method: Processing {len(competency_questions)} competency questions iteratively")
        
        # Process each CQ iteratively
        for i, cq in enumerate(competency_questions, 1):
            if self.verbatim:
                print(f"\n{'='*60}")
                print(f"[{i}/{len(competency_questions)}] Processing CQ: {cq}")
                print(f"{'='*60}")
            
            # Update ontology with new CQ
            ontology = self._process_single_cq(cq, ontology, i)
            ontology["metadata"]["processed_cqs"].append(cq)
            
            if self.verbatim:
                print(f"✅ Current ontology: {len(ontology['entity_types'])} entities, "
                      f"{len(ontology['relation_types'])} relations")
        
        return ontology
    
    def _process_single_cq(self, cq: str, current_ontology: Dict, iteration: int) -> Dict:
        """
        Process a single competency question and update ontology.
        
        Args:
            cq: Competency question
            current_ontology: Current state of ontology
            iteration: Question number
        
        Returns:
            Dict: Updated ontology
        """
        # Prepare prompt with current ontology context
        system_prompt = """You are an expert ontology engineer. Given a competency question and the current ontology, 
analyze the question and propose updates to the ontology.

Your task:
1. Identify new entity types (classes) needed
2. Identify new properties for entities
3. Identify new relations between entities
4. Ensure consistency with existing ontology

Output JSON format:
{
    "new_entity_types": [
        {"name": "EntityName", "properties": ["prop1", "prop2"], "description": "..."}
    ],
    "new_relation_types": [
        {"name": "relation_name", "domain": "EntityType1", "range": "EntityType2", "description": "..."}
    ],
    "reasoning": "Explanation of your decisions"
}"""
        
        user_prompt = f"""Current Ontology:
Entity Types: {json.dumps(list(current_ontology['entity_types'].keys()), indent=2)}
Relation Types: {json.dumps(list(current_ontology['relation_types'].keys()), indent=2)}

Competency Question #{iteration}: {cq}

Analyze this question and propose ontology updates."""
        
        # Generate LLM response
        response = self.generate_fn(system_prompt=system_prompt, prompt=user_prompt)
        
        # Parse response
        try:
            updates = self._parse_llm_response(response)
            
            # Integrate updates
            current_ontology = self._integrate_updates(current_ontology, updates)
            
            if self.verbatim:
                print(f"📝 Reasoning: {updates.get('reasoning', 'N/A')}")
                print(f"➕ Added {len(updates.get('new_entity_types', []))} entity types")
                print(f"➕ Added {len(updates.get('new_relation_types', []))} relations")
        
        except Exception as e:
            if self.verbatim:
                print(f"⚠️  Error processing CQ: {e}")
        
        return current_ontology
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        # Try to extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            return json.loads(json_str)
        
        return {"new_entity_types": [], "new_relation_types": [], "reasoning": ""}
    
    def _integrate_updates(self, ontology: Dict, updates: Dict) -> Dict:
        """Integrate LLM updates into ontology."""
        # Add new entity types
        for entity in updates.get("new_entity_types", []):
            name = entity.get("name")
            if name and name not in ontology["entity_types"]:
                ontology["entity_types"][name] = {
                    "name": name,
                    "properties": entity.get("properties", []),
                    "description": entity.get("description", "")
                }
        
        # Add new relation types
        for relation in updates.get("new_relation_types", []):
            name = relation.get("name")
            if name and name not in ontology["relation_types"]:
                ontology["relation_types"][name] = {
                    "name": name,
                    "domain": relation.get("domain", "Thing"),
                    "range": relation.get("range", "Thing"),
                    "description": relation.get("description", "")
                }
        
        return ontology


class MemorylessCQbyCQGenerator:
    """
    Memoryless CQbyCQ Method
    
    Process each competency question independently without maintaining context
    from previous questions. Final ontology is merged from all individual results.
    
    Based on: "Ontology Generation using Large Language Models"
    
    Algorithm:
    1. For each CQ independently:
       - Send CQ to LLM (no previous context)
       - LLM generates ontology fragment
    2. Merge all fragments into single ontology
    3. Resolve conflicts and duplicates
    
    Advantages:
    - No error accumulation
    - Parallelizable (can process CQs concurrently)
    - More robust to individual failures
    
    Disadvantages:
    - May miss cross-question relationships
    - Requires sophisticated merging strategy
    - Potential inconsistencies between fragments
    """
    
    def __init__(self, generate_fn=None, verbatim: bool = False):
        """
        Initialize Memoryless CQbyCQ generator.
        
        Args:
            generate_fn: LLM generation function
            verbatim: Print detailed logs
        """
        if generate_fn is None:
            generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
        self.generate_fn = generate_fn
        self.verbatim = verbatim
    
    def generate_ontology(self, competency_questions: List[str]) -> Dict:
        """
        Generate ontology using Memoryless CQbyCQ method.
        
        Args:
            competency_questions: List of competency questions
        
        Returns:
            Dict: Merged ontology from all CQs
        """
        if self.verbatim:
            print(f"🔬 Memoryless CQbyCQ: Processing {len(competency_questions)} CQs independently")
        
        # Process each CQ independently
        ontology_fragments = []
        
        for i, cq in enumerate(competency_questions, 1):
            if self.verbatim:
                print(f"\n[{i}/{len(competency_questions)}] Processing: {cq}")
            
            fragment = self._process_independent_cq(cq, i)
            ontology_fragments.append(fragment)
            
            if self.verbatim:
                print(f"✅ Fragment {i}: {len(fragment['entity_types'])} entities, "
                      f"{len(fragment['relation_types'])} relations")
        
        # Merge all fragments
        if self.verbatim:
            print(f"\n{'='*60}")
            print("🔀 Merging {len(ontology_fragments)} ontology fragments")
            print(f"{'='*60}")
        
        merged_ontology = self._merge_fragments(ontology_fragments, competency_questions)
        
        if self.verbatim:
            print(f"\n✅ Final ontology: {len(merged_ontology['entity_types'])} entities, "
                  f"{len(merged_ontology['relation_types'])} relations")
        
        return merged_ontology
    
    def _process_independent_cq(self, cq: str, cq_number: int) -> Dict:
        """
        Process a single CQ without context from other CQs.
        
        Args:
            cq: Competency question
            cq_number: Question number for tracking
        
        Returns:
            Dict: Ontology fragment from this CQ
        """
        system_prompt = """You are an expert ontology engineer. Given a competency question,
extract the ontology elements (classes, properties, relations) needed to answer it.

Output JSON format:
{
    "entity_types": [
        {"name": "EntityName", "properties": ["prop1", "prop2"], "description": "..."}
    ],
    "relation_types": [
        {"name": "relation_name", "domain": "EntityType1", "range": "EntityType2", "description": "..."}
    ]
}"""
        
        user_prompt = f"""Competency Question: {cq}

Extract the ontology elements needed to answer this question."""
        
        # Generate LLM response
        response = self.generate_fn(system_prompt=system_prompt, prompt=user_prompt)
        
        # Parse response
        try:
            fragment = self._parse_llm_response(response)
            fragment["metadata"] = {"cq_number": cq_number, "cq": cq}
            return fragment
        except Exception as e:
            if self.verbatim:
                print(f"⚠️  Error processing CQ {cq_number}: {e}")
            return {"entity_types": [], "relation_types": [], "metadata": {"cq_number": cq_number}}
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            
            # Convert list format to dict format
            result = {"entity_types": {}, "relation_types": {}}
            
            for entity in parsed.get("entity_types", []):
                name = entity.get("name")
                if name:
                    result["entity_types"][name] = entity
            
            for relation in parsed.get("relation_types", []):
                name = relation.get("name")
                if name:
                    result["relation_types"][name] = relation
            
            return result
        
        return {"entity_types": {}, "relation_types": {}}
    
    def _merge_fragments(self, fragments: List[Dict], cqs: List[str]) -> Dict:
        """
        Merge ontology fragments with conflict resolution.
        
        Args:
            fragments: List of ontology fragments
            cqs: Original competency questions
        
        Returns:
            Dict: Merged ontology
        """
        merged = {
            "entity_types": {},
            "relation_types": {},
            "metadata": {
                "method": "Memoryless CQbyCQ",
                "total_fragments": len(fragments),
                "competency_questions": cqs
            }
        }
        
        # Merge entity types
        for fragment in fragments:
            for name, entity in fragment.get("entity_types", {}).items():
                if name not in merged["entity_types"]:
                    merged["entity_types"][name] = entity
                else:
                    # Merge properties
                    existing_props = set(merged["entity_types"][name].get("properties", []))
                    new_props = set(entity.get("properties", []))
                    merged["entity_types"][name]["properties"] = list(existing_props | new_props)
        
        # Merge relation types
        for fragment in fragments:
            for name, relation in fragment.get("relation_types", {}).items():
                if name not in merged["relation_types"]:
                    merged["relation_types"][name] = relation
                else:
                    # Keep first occurrence for domain/range
                    # Could implement more sophisticated conflict resolution
                    pass
        
        return merged


class OntogeniaGenerator:
    """
    Ontogenia Method
    
    All-at-once approach that processes all competency questions simultaneously
    in a single LLM call to generate a comprehensive, globally consistent ontology.
    
    Based on: "Ontology Generation using Large Language Models"
    
    Algorithm:
    1. Present all CQs to LLM in one prompt
    2. LLM analyzes all questions holistically
    3. Generate complete, consistent ontology in single pass
    
    Advantages:
    - Global consistency
    - Can identify cross-CQ patterns
    - Simplest implementation
    - Fastest (single LLM call)
    
    Disadvantages:
    - Limited by context window size
    - No iterative refinement
    - All-or-nothing approach
    """
    
    def __init__(self, generate_fn=None, verbatim: bool = False):
        """
        Initialize Ontogenia generator.
        
        Args:
            generate_fn: LLM generation function
            verbatim: Print detailed logs
        """
        if generate_fn is None:
            generate_fn = get_generate_fn("openai", config={"model": "gpt-4"})
        self.generate_fn = generate_fn
        self.verbatim = verbatim
    
    def generate_ontology(self, competency_questions: List[str]) -> Dict:
        """
        Generate ontology using Ontogenia (all-at-once) method.
        
        Args:
            competency_questions: List of competency questions
        
        Returns:
            Dict: Complete ontology
        """
        if self.verbatim:
            print(f"🔬 Ontogenia Method: Processing all {len(competency_questions)} CQs simultaneously")
        
        # Create comprehensive prompt with all CQs
        system_prompt = """You are an expert ontology engineer. Given a set of competency questions,
design a comprehensive ontology that can answer all questions.

Your task:
1. Analyze all questions holistically
2. Identify entity types (classes) needed
3. Define properties for each entity type
4. Define relations between entity types
5. Ensure global consistency across all elements

Output JSON format:
{
    "entity_types": {
        "EntityName": {
            "name": "EntityName",
            "properties": ["prop1", "prop2", "prop3"],
            "description": "Description of entity type"
        }
    },
    "relation_types": {
        "relation_name": {
            "name": "relation_name",
            "domain": "DomainEntityType",
            "range": "RangeEntityType",
            "description": "Description of relation"
        }
    }
}"""
        
        # Format all CQs in user prompt
        cq_text = "\n".join([f"{i+1}. {cq}" for i, cq in enumerate(competency_questions)])
        
        user_prompt = f"""Competency Questions (Total: {len(competency_questions)}):

{cq_text}

Design a comprehensive ontology that covers all these competency questions."""
        
        if self.verbatim:
            print(f"\n📤 Sending {len(competency_questions)} CQs to LLM...")
        
        # Generate ontology in single LLM call
        response = self.generate_fn(system_prompt=system_prompt, prompt=user_prompt)
        
        if self.verbatim:
            print(f"📥 Received response, parsing ontology...")
        
        # Parse response
        try:
            ontology = self._parse_llm_response(response)
            ontology["metadata"] = {
                "method": "Ontogenia",
                "total_cqs": len(competency_questions),
                "competency_questions": competency_questions
            }
            
            if self.verbatim:
                print(f"\n✅ Generated ontology:")
                print(f"   Entity types: {len(ontology['entity_types'])}")
                print(f"   Relation types: {len(ontology['relation_types'])}")
                
                if ontology['entity_types']:
                    print(f"\n   Entities: {', '.join(list(ontology['entity_types'].keys())[:10])}")
                if ontology['relation_types']:
                    print(f"   Relations: {', '.join(list(ontology['relation_types'].keys())[:10])}")
            
            return ontology
            
        except Exception as e:
            if self.verbatim:
                print(f"⚠️  Error parsing response: {e}")
                print(f"Response preview: {response[:500]}")
            
            return {
                "entity_types": {},
                "relation_types": {},
                "metadata": {
                    "method": "Ontogenia",
                    "error": str(e),
                    "competency_questions": competency_questions
                }
            }
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        # Remove markdown code blocks if present
        response = response.replace("```json", "").replace("```", "")
        
        # Find JSON boundaries
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            ontology = json.loads(json_str)
            
            # Ensure proper structure
            if "entity_types" not in ontology:
                ontology["entity_types"] = {}
            if "relation_types" not in ontology:
                ontology["relation_types"] = {}
            
            return ontology
        
        return {"entity_types": {}, "relation_types": {}}

####################################################################
# Bottom-up Triples-based Ontology Generation Methods
####################################################################


class BottomUpOntologyInducer:
    """Induce ontology from triples (bottom-up approach)."""
    
    def __init__(self):
        self.analyzer = TripleAnalyzer()
    
    def induce_ontology_from_triples(self, triples: List[Dict],
                                    min_frequency: int = 2,
                                    verbatim: bool = False) -> Dict:
        """
        Induce ontology/schema from triples.
        
        Args:
            triples (List[Dict]): List of triple dictionaries
            min_frequency (int): Minimum frequency threshold
            verbatim (bool): Print verbose output
        
        Returns:
            Dict: Induced ontology with entity and relation types
        """
        # Analyze triples
        analysis = self.analyzer.analyze_triples(triples)
        
        if verbatim:
            print(f"📊 Triple Analysis:")
            print(f"   Total triples: {analysis['total_triples']}")
            print(f"   Unique entities: {analysis['unique_entities_count']}")
            print(f"   Unique relations: {analysis['unique_relations_count']}")
        
        # Infer entity types
        entity_types = self.analyzer.infer_entity_types(
            triples, 
            min_frequency=min_frequency,
            verbatim=verbatim
        )
        
        # Infer relation types
        relation_types = self.analyzer.infer_relation_types(
            triples,
            min_frequency=min_frequency,
            verbatim=verbatim
        )
        
        # Build ontology
        ontology = {
            "entity_types": {
                name: asdict(entity) 
                for name, entity in entity_types.items()
            },
            "relation_types": {
                name: asdict(relation)
                for name, relation in relation_types.items()
            },
            "statistics": {
                "total_entities": analysis["unique_entities_count"],
                "total_relations": analysis["unique_relations_count"],
                "total_triples": analysis["total_triples"],
            }
        }
        
        return ontology


class LLMBasedBottomUpGenerator:
    """
    LLM-based Bottom-Up ontology generation from triples.
    
    Implements the paper "Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation"
    using three strategies:
    1. Instance-level: Extract types from individual instances
    2. Pattern-based: Identify patterns across multiple instances
    3. Semantic: Use LLM to understand semantic relationships
    """
    
    def __init__(self, generate_fn=None, verbatim: bool = False):
        """
        Initialize LLM-based Bottom-Up generator.
        
        Args:
            generate_fn: LLM generation function
            verbatim: Print detailed logs
        """
        self.generate_fn = generate_fn or get_generate_fn("openai", config={"model": "gpt-4"})
        self.verbatim = verbatim
        self.analyzer = TripleAnalyzer()
    
    def generate_ontology(self, triples: List[Dict], 
                         strategy: str = 'semantic',
                         sample_size: int = None,
                         verbatim: bool = None) -> Dict:
        """
        Generate ontology from triples using LLM-based Bottom-Up approach.
        
        Args:
            triples (List[Dict]): List of triples with keys:
                ["node_1", "node_2", "edge", "node_1_type", "node_2_type"]
            strategy (str): Generation strategy - 'instance', 'pattern', or 'semantic'
            sample_size (int): Number of triples to sample (None = use all)
            verbatim (bool): Print detailed logs (uses default if None)
        
        Returns:
            Dict: Generated ontology with entity and relation types
        """
        verbatim = verbatim if verbatim is not None else self.verbatim
        
        if verbatim:
            print("="*70)
            print(f"🔬 LLM-Based Bottom-Up Ontology Generation ({strategy})")
            print("="*70)
            print(f"Total triples: {len(triples)}")
        
        # Sample triples if needed
        triples_to_analyze = triples
        if sample_size and len(triples) > sample_size:
            import random
            triples_to_analyze = random.sample(triples, sample_size)
            if verbatim:
                print(f"Sampled {sample_size} triples for analysis")
        
        # Analyze triples
        analysis = self.analyzer.analyze_triples(triples_to_analyze)
        
        if verbatim:
            print(f"\n📊 Analysis Results:")
            print(f"   Unique entities: {analysis['unique_entities_count']}")
            print(f"   Unique relations: {analysis['unique_relations_count']}")
        
        # Generate ontology based on strategy
        if strategy == 'instance':
            return self._generate_instance_level(triples_to_analyze, analysis, verbatim)
        elif strategy == 'pattern':
            return self._generate_pattern_based(triples_to_analyze, analysis, verbatim)
        elif strategy == 'semantic':
            return self._generate_semantic_level(triples_to_analyze, analysis, verbatim)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'instance', 'pattern', or 'semantic'")
    
    def _generate_instance_level(self, triples: List[Dict], 
                                analysis: Dict,
                                verbatim: bool = False) -> Dict:
        """
        Instance-level extraction: Extract types from individual instances.
        
        Strategy: For each unique instance, extract its type and properties.
        """
        if verbatim:
            print("\n1️⃣ Instance-Level Extraction")
            print("-" * 70)
        
        # Group instances by their provided types
        instances_by_type = defaultdict(list)
        properties_by_instance = defaultdict(set)
        
        for triple in triples:
            node_1 = triple.get("node_1", "")
            node_2 = triple.get("node_2", "")
            edge = triple.get("edge", "")
            node_1_type = triple.get("node_1_type", "Unknown")
            node_2_type = triple.get("node_2_type", "Unknown")
            
            # Track instances by type
            instances_by_type[node_1_type].append(node_1)
            instances_by_type[node_2_type].append(node_2)
            
            # Track properties
            properties_by_instance[node_1].add(edge)
            properties_by_instance[node_2].add(edge)
        
        # Use LLM to validate and enhance entity types
        entity_types = {}
        for entity_type, instances in instances_by_type.items():
            if entity_type == "Unknown":
                continue
            
            unique_instances = list(set(instances))[:5]
            system_prompt = """You are an ontology expert. Analyze the following instances 
and provide a comprehensive entity type definition."""
            
            user_prompt = f"""Given these instances of type '{entity_type}':
{json.dumps(unique_instances)}

Provide:
1. A precise definition of this entity type
2. Common properties/attributes
3. Confidence score (0-1)

Respond in JSON format with keys: definition, properties, confidence"""
            
            try:
                response = self.generate_fn(system_prompt=system_prompt, prompt=user_prompt)
                llm_result = self._extract_json(response)
                
                entity_types[entity_type] = {
                    "name": entity_type,
                    "definition": llm_result.get("definition", f"Type {entity_type}"),
                    "properties": llm_result.get("properties", []),
                    "confidence": float(llm_result.get("confidence", 0.8)),
                    "instances": unique_instances,
                    "count": len(set(instances))
                }
                
                if verbatim:
                    print(f"✅ {entity_type}: {len(set(instances))} instances")
            except Exception as e:
                if verbatim:
                    print(f"⚠️  {entity_type}: Using fallback (error: {str(e)[:50]})")
                
                # Fallback
                entity_types[entity_type] = {
                    "name": entity_type,
                    "definition": f"Entity type: {entity_type}",
                    "properties": list(set(prop for inst in unique_instances 
                                          for prop in properties_by_instance.get(inst, []))),
                    "confidence": 0.6,
                    "instances": unique_instances,
                    "count": len(set(instances))
                }
        
        # Infer relation types using basic analysis
        relation_types = self.analyzer.infer_relation_types(triples, verbatim=False)
        
        return {
            "entity_types": entity_types,
            "relation_types": {
                name: asdict(rel) for name, rel in relation_types.items()
            },
            "metadata": {
                "method": "LLM-based_instance_level",
                "strategy": "instance",
                "total_triples": len(triples)
            }
        }
    
    def _generate_pattern_based(self, triples: List[Dict],
                               analysis: Dict,
                               verbatim: bool = False) -> Dict:
        """
        Pattern-based extraction: Identify patterns across instances.
        
        Strategy: Group similar relations and identify common patterns.
        """
        if verbatim:
            print("\n2️⃣ Pattern-Based Extraction")
            print("-" * 70)
        
        # Group triples by relation
        triples_by_relation = defaultdict(list)
        for triple in triples:
            edge = triple.get("edge", "")
            triples_by_relation[edge].append(triple)
        
        # Use LLM to identify domain-range patterns
        relation_types = {}
        for edge, edge_triples in triples_by_relation.items():
            # Sample domain-range pairs
            domain_range_samples = []
            for triple in edge_triples[:3]:  # Sample 3 examples
                domain_range_samples.append({
                    "source": triple.get("node_1_type", "Unknown"),
                    "target": triple.get("node_2_type", "Unknown"),
                    "example": f"{triple.get('node_1')} → {triple.get('node_2')}"
                })
            
            system_prompt = """You are an ontology expert. Analyze relation patterns 
and provide domain-range definitions."""
            
            user_prompt = f"""Analyze this relation '{edge}' with domain-range examples:
{json.dumps(domain_range_samples)}

Identify:
1. Most likely domain entity type
2. Most likely range entity type
3. Semantic meaning of the relation
4. Confidence score (0-1)

Respond in JSON with: domain, range, semantic_meaning, confidence"""
            
            try:
                response = self.generate_fn(system_prompt=system_prompt, prompt=user_prompt)
                llm_result = self._extract_json(response)
                
                relation_types[edge] = {
                    "name": edge,
                    "domain": llm_result.get("domain", "Thing"),
                    "range": llm_result.get("range", "Thing"),
                    "semantic_meaning": llm_result.get("semantic_meaning", edge),
                    "confidence": float(llm_result.get("confidence", 0.7)),
                    "frequency": len(edge_triples)
                }
                
                if verbatim:
                    print(f"✅ {edge}: {relation_types[edge]['domain']} → {relation_types[edge]['range']}")
            except Exception as e:
                if verbatim:
                    print(f"⚠️  {edge}: Using fallback")
                
                # Fallback
                domain_range = max(
                    [(t.get("node_1_type", "Unknown"), t.get("node_2_type", "Unknown"))
                     for t in edge_triples],
                    key=lambda x: sum(1 for t in edge_triples 
                                     if t.get("node_1_type") == x[0] and t.get("node_2_type") == x[1]),
                    default=("Thing", "Thing")
                )
                
                relation_types[edge] = {
                    "name": edge,
                    "domain": domain_range[0],
                    "range": domain_range[1],
                    "semantic_meaning": edge,
                    "confidence": 0.5,
                    "frequency": len(edge_triples)
                }
        
        # Infer entity types
        entity_types = self.analyzer.infer_entity_types(triples, verbatim=False)
        
        return {
            "entity_types": {
                name: asdict(ent) for name, ent in entity_types.items()
            },
            "relation_types": relation_types,
            "metadata": {
                "method": "LLM-based_pattern",
                "strategy": "pattern",
                "total_triples": len(triples)
            }
        }
    
    def _generate_semantic_level(self, triples: List[Dict],
                                analysis: Dict,
                                verbatim: bool = False) -> Dict:
        """
        Semantic-level extraction: Deep semantic understanding using LLM.
        
        Strategy: Comprehensive LLM analysis of all triples for semantic relationships.
        """
        if verbatim:
            print("\n3️⃣ Semantic-Level Extraction")
            print("-" * 70)
        
        # Prepare triple descriptions
        triple_descriptions = []
        for triple in triples[:20]:  # Use first 20 triples
            triple_descriptions.append(
                f"{triple.get('node_1')} ({triple.get('node_1_type', 'Unknown')}) "
                f"--{triple.get('edge')}--> "
                f"{triple.get('node_2')} ({triple.get('node_2_type', 'Unknown')})"
            )
        
        system_prompt = """You are an expert ontologist with deep understanding of semantic web technologies.
Analyze the provided knowledge graph triples and extract comprehensive ontology definitions."""
        
        user_prompt = f"""Analyze these {len(triple_descriptions)} knowledge graph triples:

{chr(10).join(triple_descriptions)}

For this knowledge graph, provide:
1. All entity types with definitions and properties
2. All relation types with domain, range, and semantic meaning
3. Overall domain/topic
4. Confidence scores

Respond in JSON format:
{{
  "domain_topic": "...",
  "entity_types": {{
    "TypeName": {{
      "definition": "...",
      "properties": ["prop1", "prop2"],
      "confidence": 0.9
    }},
    ...
  }},
  "relation_types": {{
    "relationName": {{
      "domain": "...",
      "range": "...",
      "semantic_meaning": "...",
      "confidence": 0.85
    }},
    ...
  }}
}}"""
        
        try:
            response = self.generate_fn(system_prompt=system_prompt, prompt=user_prompt)
            llm_result = self._extract_json(response)
            
            # Process entity types
            entity_types = {}
            for entity_name, entity_info in llm_result.get("entity_types", {}).items():
                entity_types[entity_name] = {
                    "name": entity_name,
                    "definition": entity_info.get("definition", f"Type {entity_name}"),
                    "properties": entity_info.get("properties", []),
                    "confidence": float(entity_info.get("confidence", 0.8))
                }
            
            # Process relation types
            relation_types = {}
            for rel_name, rel_info in llm_result.get("relation_types", {}).items():
                relation_types[rel_name] = {
                    "name": rel_name,
                    "domain": rel_info.get("domain", "Thing"),
                    "range": rel_info.get("range", "Thing"),
                    "semantic_meaning": rel_info.get("semantic_meaning", rel_name),
                    "confidence": float(rel_info.get("confidence", 0.7))
                }
            
            if verbatim:
                print(f"✅ Extracted {len(entity_types)} entity types, {len(relation_types)} relation types")
                print(f"   Domain/Topic: {llm_result.get('domain_topic', 'Unknown')}")
        
        except Exception as e:
            if verbatim:
                print(f"⚠️  LLM semantic analysis failed, using pattern-based fallback")
            
            # Fallback to pattern-based
            return self._generate_pattern_based(triples, analysis, verbatim=False)
        
        return {
            "entity_types": entity_types,
            "relation_types": relation_types,
            "metadata": {
                "method": "LLM-based_semantic",
                "strategy": "semantic",
                "domain_topic": llm_result.get("domain_topic", "Unknown"),
                "total_triples": len(triples)
            }
        }
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON in the response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except:
            pass
        
        # Fallback: return empty dict
        return {}

#####################################################################
# Ontology Merging and Serialization Methods
####################################################################

class OntologyMerger:
    """Merge ontologies from different sources."""
    
    @staticmethod
    def merge_ontologies(ontologies: List[Dict],
                        strategy: str = "union",
                        verbatim: bool = False) -> Dict:
        """
        Merge multiple ontologies.
        
        Args:
            ontologies (List[Dict]): List of ontology dictionaries
            strategy (str): Merge strategy - "union" or "intersection"
            verbatim (bool): Print verbose output
        
        Returns:
            Dict: Merged ontology
        """
        if not ontologies:
            return {"entity_types": {}, "relation_types": {}}
        
        merged = {
            "entity_types": {},
            "relation_types": {},
        }
        
        if strategy == "union":
            # Take union of all types
            for onto in ontologies:
                merged["entity_types"].update(onto.get("entity_types", {}))
                merged["relation_types"].update(onto.get("relation_types", {}))
        
        elif strategy == "intersection":
            # Take intersection of all types
            if ontologies:
                entity_sets = [set(onto.get("entity_types", {}).keys()) for onto in ontologies]
                relation_sets = [set(onto.get("relation_types", {}).keys()) for onto in ontologies]
                
                common_entities = set.intersection(*entity_sets) if entity_sets else set()
                common_relations = set.intersection(*relation_sets) if relation_sets else set()
                
                for onto in ontologies:
                    for ent in common_entities:
                        if ent in onto.get("entity_types", {}):
                            merged["entity_types"][ent] = onto["entity_types"][ent]
                    
                    for rel in common_relations:
                        if rel in onto.get("relation_types", {}):
                            merged["relation_types"][rel] = onto["relation_types"][rel]
                
                if verbatim:
                    print(f"✅ Common entities: {len(common_entities)}")
                    print(f"✅ Common relations: {len(common_relations)}")
        
        return merged


class OntologySerializer:
    """Serialize and deserialize ontologies in various formats."""
    
    SUPPORTED_FORMATS = ["json", "yaml", "owl", "rdf", "turtle", "csv"]
    
    @staticmethod
    def save_ontology(ontology: Dict, 
                     filepath: Union[str, Path],
                     format: str = "json",
                     pretty: bool = True,
                     include_metadata: bool = True) -> None:
        """
        Save ontology to file in specified format.
        
        Args:
            ontology (Dict): Ontology to save
            filepath (Union[str, Path]): Output file path
            format (str): Output format - "json", "yaml", "owl", "rdf", "turtle", "csv"
            pretty (bool): Pretty print output
            include_metadata (bool): Include metadata (creation date, version, etc.)
        """
        filepath = Path(filepath)
        
        # Add metadata if requested
        if include_metadata:
            ontology = OntologySerializer._add_metadata(ontology)
        
        # Save based on format
        if format == "json":
            OntologySerializer._save_json(ontology, filepath, pretty)
        elif format == "yaml":
            OntologySerializer._save_yaml(ontology, filepath)
        elif format == "owl":
            OntologySerializer._save_owl(ontology, filepath)
        elif format == "rdf":
            OntologySerializer._save_rdf(ontology, filepath, rdf_format="xml")
        elif format == "turtle":
            OntologySerializer._save_rdf(ontology, filepath, rdf_format="turtle")
        elif format == "csv":
            OntologySerializer._save_csv(ontology, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}. Must be one of {OntologySerializer.SUPPORTED_FORMATS}")
    
    @staticmethod
    def load_ontology(filepath: Union[str, Path],
                     format: str = None) -> Dict:
        """
        Load ontology from file.
        
        Args:
            filepath (Union[str, Path]): Input file path
            format (str): Input format. If None, infer from file extension
        
        Returns:
            Dict: Loaded ontology
        """
        filepath = Path(filepath)
        
        # Infer format from extension if not provided
        if format is None:
            format = filepath.suffix.lstrip('.')
            if format == "yml":
                format = "yaml"
            elif format in ["rdf", "xml"]:
                format = "rdf"
            elif format == "ttl":
                format = "turtle"
        
        # Load based on format
        if format == "json":
            return OntologySerializer._load_json(filepath)
        elif format == "yaml":
            return OntologySerializer._load_yaml(filepath)
        elif format in ["owl", "rdf", "turtle"]:
            return OntologySerializer._load_rdf(filepath)
        elif format == "csv":
            return OntologySerializer._load_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _add_metadata(ontology: Dict) -> Dict:
        """Add metadata to ontology."""
        ontology_with_meta = ontology.copy()
        ontology_with_meta["metadata"] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "generator": "GraphConstruct.onto_generation",
            "entity_count": len(ontology.get("entity_types", {})),
            "relation_count": len(ontology.get("relation_types", {})),
        }
        return ontology_with_meta
    
    @staticmethod
    def _save_json(ontology: Dict, filepath: Path, pretty: bool = True) -> None:
        """Save as JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(ontology, f, indent=2, ensure_ascii=False)
            else:
                json.dump(ontology, f, ensure_ascii=False)
        print(f"✅ Ontology saved to {filepath} (JSON)")
    
    @staticmethod
    def _save_yaml(ontology: Dict, filepath: Path) -> None:
        """Save as YAML."""
        try:
            import yaml
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(ontology, f, default_flow_style=False, allow_unicode=True)
            print(f"✅ Ontology saved to {filepath} (YAML)")
        except ImportError:
            print("⚠️  YAML support requires 'pyyaml' package. Falling back to JSON.")
            OntologySerializer._save_json(ontology, filepath.with_suffix('.json'))
    
    @staticmethod
    def _save_owl(ontology: Dict, filepath: Path) -> None:
        """Save as OWL (Web Ontology Language)."""
        owl_content = OntologySerializer._ontology_to_owl(ontology)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(owl_content)
        print(f"✅ Ontology saved to {filepath} (OWL)")
    
    @staticmethod
    def _save_rdf(ontology: Dict, filepath: Path, rdf_format: str = "xml") -> None:
        """Save as RDF (Resource Description Framework)."""
        try:
            from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal
            
            g = Graph()
            
            # Define namespaces
            BASE = Namespace("http://example.org/ontology#")
            g.bind("base", BASE)
            g.bind("rdf", RDF)
            g.bind("rdfs", RDFS)
            g.bind("owl", OWL)
            
            # Add entity types as OWL classes
            for entity_name, entity_info in ontology.get("entity_types", {}).items():
                entity_uri = BASE[entity_name]
                g.add((entity_uri, RDF.type, OWL.Class))
                g.add((entity_uri, RDFS.label, Literal(entity_name)))
                
                # Add description if available
                description = entity_info.get("description", "")
                if description:
                    g.add((entity_uri, RDFS.comment, Literal(description)))
            
            # Add relation types as OWL object properties
            for relation_name, relation_info in ontology.get("relation_types", {}).items():
                relation_uri = BASE[relation_name]
                g.add((relation_uri, RDF.type, OWL.ObjectProperty))
                g.add((relation_uri, RDFS.label, Literal(relation_name)))
                
                # Add domain and range
                domain = relation_info.get("domain", "Thing")
                range_ = relation_info.get("range", "Thing")
                
                g.add((relation_uri, RDFS.domain, BASE[domain]))
                g.add((relation_uri, RDFS.range, BASE[range_]))
            
            # Serialize to file
            format_map = {
                "xml": "xml",
                "turtle": "turtle",
                "n3": "n3",
                "nt": "nt"
            }
            
            g.serialize(destination=str(filepath), format=format_map.get(rdf_format, "xml"))
            print(f"✅ Ontology saved to {filepath} (RDF/{rdf_format.upper()})")
            
        except ImportError:
            print("⚠️  RDF support requires 'rdflib' package. Falling back to OWL.")
            OntologySerializer._save_owl(ontology, filepath.with_suffix('.owl'))
    
    @staticmethod
    def _save_csv(ontology: Dict, filepath: Path) -> None:
        """Save as CSV (separate files for entities and relations)."""
        base_path = filepath.parent
        base_name = filepath.stem
        
        # Save entity types
        entity_data = []
        for name, info in ontology.get("entity_types", {}).items():
            entity_data.append({
                "entity_type": name,
                "properties": ", ".join(info.get("properties", [])),
                "description": info.get("description", ""),
                "frequency": info.get("frequency", 0),
                "examples": ", ".join(info.get("examples", [])[:3])
            })
        
        if entity_data:
            df_entities = pd.DataFrame(entity_data)
            entity_file = base_path / f"{base_name}_entities.csv"
            df_entities.to_csv(entity_file, index=False)
            print(f"✅ Entity types saved to {entity_file}")
        
        # Save relation types
        relation_data = []
        for name, info in ontology.get("relation_types", {}).items():
            relation_data.append({
                "relation_type": name,
                "domain": info.get("domain", ""),
                "range": info.get("range", ""),
                "frequency": info.get("frequency", 0),
                "confidence": info.get("confidence", 0.0),
                "description": info.get("description", "")
            })
        
        if relation_data:
            df_relations = pd.DataFrame(relation_data)
            relation_file = base_path / f"{base_name}_relations.csv"
            df_relations.to_csv(relation_file, index=False)
            print(f"✅ Relation types saved to {relation_file}")
    
    @staticmethod
    def _load_json(filepath: Path) -> Dict:
        """Load from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        print(f"✅ Ontology loaded from {filepath} (JSON)")
        return ontology
    
    @staticmethod
    def _load_yaml(filepath: Path) -> Dict:
        """Load from YAML."""
        try:
            import yaml
            with open(filepath, 'r', encoding='utf-8') as f:
                ontology = yaml.safe_load(f)
            print(f"✅ Ontology loaded from {filepath} (YAML)")
            return ontology
        except ImportError:
            raise ImportError("YAML support requires 'pyyaml' package")
    
    @staticmethod
    def _load_rdf(filepath: Path) -> Dict:
        """Load from RDF/OWL."""
        try:
            from rdflib import Graph, RDF, RDFS, OWL
            
            g = Graph()
            g.parse(str(filepath))
            
            ontology = {
                "entity_types": {},
                "relation_types": {}
            }
            
            # Extract classes (entity types)
            for s in g.subjects(RDF.type, OWL.Class):
                name = str(s).split('#')[-1].split('/')[-1]
                label = g.value(s, RDFS.label)
                comment = g.value(s, RDFS.comment)
                
                ontology["entity_types"][name] = {
                    "name": name,
                    "description": str(comment) if comment else "",
                    "properties": []
                }
            
            # Extract object properties (relation types)
            for s in g.subjects(RDF.type, OWL.ObjectProperty):
                name = str(s).split('#')[-1].split('/')[-1]
                domain = g.value(s, RDFS.domain)
                range_ = g.value(s, RDFS.range)
                
                ontology["relation_types"][name] = {
                    "name": name,
                    "domain": str(domain).split('#')[-1].split('/')[-1] if domain else "Thing",
                    "range": str(range_).split('#')[-1].split('/')[-1] if range_ else "Thing"
                }
            
            print(f"✅ Ontology loaded from {filepath} (RDF/OWL)")
            return ontology
            
        except ImportError:
            raise ImportError("RDF support requires 'rdflib' package")
    
    @staticmethod
    def _load_csv(filepath: Path) -> Dict:
        """Load from CSV."""
        base_path = filepath.parent
        base_name = filepath.stem
        
        ontology = {
            "entity_types": {},
            "relation_types": {}
        }
        
        # Load entity types
        entity_file = base_path / f"{base_name}_entities.csv"
        if entity_file.exists():
            df_entities = pd.read_csv(entity_file)
            for _, row in df_entities.iterrows():
                name = row["entity_type"]
                ontology["entity_types"][name] = {
                    "name": name,
                    "properties": row.get("properties", "").split(", ") if row.get("properties") else [],
                    "description": row.get("description", ""),
                    "frequency": int(row.get("frequency", 0)),
                    "examples": row.get("examples", "").split(", ") if row.get("examples") else []
                }
        
        # Load relation types
        relation_file = base_path / f"{base_name}_relations.csv"
        if relation_file.exists():
            df_relations = pd.read_csv(relation_file)
            for _, row in df_relations.iterrows():
                name = row["relation_type"]
                ontology["relation_types"][name] = {
                    "name": name,
                    "domain": row.get("domain", "Thing"),
                    "range": row.get("range", "Thing"),
                    "frequency": int(row.get("frequency", 0)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "description": row.get("description", "")
                }
        
        print(f"✅ Ontology loaded from CSV files")
        return ontology
    
    @staticmethod
    def _ontology_to_owl(ontology: Dict) -> str:
        """Convert ontology to OWL format (simple XML representation)."""
        owl_lines = [
            '<?xml version="1.0"?>',
            '<rdf:RDF xmlns="http://example.org/ontology#"',
            '     xml:base="http://example.org/ontology"',
            '     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '     xmlns:owl="http://www.w3.org/2002/07/owl#"',
            '     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">',
            '',
            '    <owl:Ontology rdf:about="http://example.org/ontology"/>',
            ''
        ]
        
        # Add entity types as OWL classes
        for entity_name, entity_info in ontology.get("entity_types", {}).items():
            owl_lines.append(f'    <owl:Class rdf:about="#{entity_name}">')
            owl_lines.append(f'        <rdfs:label>{entity_name}</rdfs:label>')
            
            description = entity_info.get("description", "")
            if description:
                owl_lines.append(f'        <rdfs:comment>{description}</rdfs:comment>')
            
            owl_lines.append('    </owl:Class>')
            owl_lines.append('')
        
        # Add relation types as OWL object properties
        for relation_name, relation_info in ontology.get("relation_types", {}).items():
            owl_lines.append(f'    <owl:ObjectProperty rdf:about="#{relation_name}">')
            owl_lines.append(f'        <rdfs:label>{relation_name}</rdfs:label>')
            
            domain = relation_info.get("domain", "Thing")
            range_ = relation_info.get("range", "Thing")
            
            owl_lines.append(f'        <rdfs:domain rdf:resource="#{domain}"/>')
            owl_lines.append(f'        <rdfs:range rdf:resource="#{range_}"/>')
            owl_lines.append('    </owl:ObjectProperty>')
            owl_lines.append('')
        
        owl_lines.append('</rdf:RDF>')
        
        return '\n'.join(owl_lines)

####################################################################
# Convenience Functions for New Methods
####################################################################



def generate_ontology_from_questions(questions: List[str],
                                    generate_fn=None,
                                    method: str = 'pattern',
                                    verbatim: bool = False) -> Dict:
    """
    Convenience function to generate ontology from competency questions.
    
    Args:
        questions (List[str]): Competency questions
        generate_fn: Optional LLM generation function for enhancement
        method (str): Extraction method - 'pattern', 'cqbycq', 'memoryless', or 'ontogenia'
        verbatim (bool): Print verbose output
    
    Returns:
        Dict: Generated ontology
    """
    extractor = TopDownOntologyExtractor(generate_fn, method=method)
    ontology = extractor.extract_from_competency_questions(questions, verbatim)
    return ontology



def compare_cq_methods(competency_questions: List[str],
                       generate_fn=None,
                       methods: List[str] = None,
                       verbatim: bool = False) -> Dict:
    """
    Compare ontology generation methods.
    
    Args:
        competency_questions: List of competency questions
        generate_fn: LLM generation function
        methods: List of methods to compare - defaults to all ['pattern', 'cqbycq', 'memoryless', 'ontogenia']
        verbatim: Print detailed logs
    
    Returns:
        Dict: Comparison results with all selected ontologies
    
    Example:
        >>> questions = ["Who works where?", "What happened when?"]
        >>> results = compare_cq_methods(questions)
        >>> print(f"CQbyCQ: {len(results['cqbycq']['entity_types'])} entities")
        >>> print(f"Memoryless: {len(results['memoryless']['entity_types'])} entities")
        
        >>> # Compare only specific methods
        >>> results = compare_cq_methods(questions, methods=['pattern', 'cqbycq'])
    """
    if methods is None:
        methods = ['pattern', 'cqbycq', 'memoryless', 'ontogenia']
    
    if verbatim:
        print("="*70)
        print(f"COMPARING ONTOLOGY GENERATION METHODS: {', '.join([m.upper() for m in methods])}")
        print("="*70)
    
    results = {}
    
    for method in methods:
        if verbatim:
            print("\n" + "="*70)
            print(f"METHOD: {method.upper()}")
            print("="*70)
        
        if method == 'pattern':
            extractor = TopDownOntologyExtractor(generate_fn, method='pattern')
            results['pattern'] = extractor.extract_from_competency_questions(
                competency_questions, verbatim
            )
        elif method == 'cqbycq':
            results['cqbycq'] = generate_ontology_from_questions(
                competency_questions, generate_fn=generate_fn, method='cqbycq', verbatim=verbatim
            )
        elif method == 'memoryless':
            results['memoryless'] = generate_ontology_from_questions(
                competency_questions, generate_fn=generate_fn, method='memoryless', verbatim=verbatim
            )
        elif method == 'ontogenia':
            results['ontogenia'] = generate_ontology_from_questions(
                competency_questions, generate_fn=generate_fn, method='ontogenia', verbatim=verbatim
            )
    
    # Add comparison stats
    stats = {}
    for method in methods:
        onto = results.get(method, {})
        stats[f"{method}_entities"] = len(onto.get('entity_types', {}))
        stats[f"{method}_relations"] = len(onto.get('relation_types', {}))
    
    # Summary
    if verbatim:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        for method in methods:
            onto = results.get(method, {})
            print(f"\n{method.upper()}:")
            print(f"  Entity types: {len(onto.get('entity_types', {}))}")
            print(f"  Relation types: {len(onto.get('relation_types', {}))}")
            if onto.get('entity_types'):
                entities = list(onto['entity_types'].keys())[:5]
                print(f"  Sample entities: {', '.join(entities)}")
    
    results['comparison_stats'] = stats
    return results

def generate_ontology_from_triples(triples: List[Dict],
                                  min_frequency: int = 2,
                                  method: str = 'frequency',
                                  verbatim: bool = False,
                                  **kwargs) -> Dict:
    """
    Generate ontology from triples (bottom-up approach).
    
    Args:
        triples (List[Dict]): List of triples with keys ["node_1", "node_2", "edge", ...]
        min_frequency (int): Minimum frequency threshold for entity/relation types
        method (str): Type inference method - 'frequency', 'pattern', 'llm', 'hybrid'
        verbatim (bool): Print verbose output
        **kwargs: Additional arguments passed to BottomUpOntologyInducer
    
    Returns:
        Dict: Generated ontology with entity and relation types
    
    Example:
        >>> triples = [
        ...     {"node_1": "Alice", "node_2": "Bob", "edge": "knows"},
        ...     {"node_1": "Bob", "node_2": "Company X", "edge": "works_for"}
        ... ]
        >>> ontology = generate_ontology_from_triples(triples, method='frequency')
    """
    inducer = BottomUpOntologyInducer()
    ontology = inducer.induce_ontology_from_triples(
        triples, min_frequency=min_frequency, verbatim=verbatim, **kwargs
    )
    
    # Add method metadata
    ontology['metadata'] = {
        'method': 'bottom_up',
        'inference_method': method,
        'min_frequency': min_frequency
    }
    
    return ontology


def generate_ontology_llm_bottomup(triples: List[Dict],
                                   generate_fn=None,
                                   strategy: str = 'semantic',
                                   sample_size: int = None,
                                   verbatim: bool = False,
                                   **kwargs) -> Dict:
    """
    Generate ontology from triples using LLM-based Bottom-Up approach.
    
    Implements "Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation"
    with three extraction strategies:
    - 'instance': Extract types from individual instances
    - 'pattern': Identify patterns across multiple instances
    - 'semantic': Deep semantic understanding using LLM
    
    Args:
        triples (List[Dict]): List of triples with keys ["node_1", "node_2", "edge", "node_1_type", "node_2_type"]
        generate_fn: LLM generation function (uses default OpenAI GPT-4 if None)
        strategy (str): Extraction strategy - 'instance', 'pattern', or 'semantic'
        sample_size (int): Number of triples to sample (None = use all)
        verbatim (bool): Print detailed logs
        **kwargs: Additional arguments
    
    Returns:
        Dict: Generated ontology with entity and relation types
    
    Example:
        >>> triples = [
        ...     {"node_1": "Alice", "node_2": "Bob", "edge": "knows", "node_1_type": "Person", "node_2_type": "Person"},
        ...     {"node_1": "Bob", "node_2": "MIT", "edge": "works_for", "node_1_type": "Person", "node_2_type": "Organization"}
        ... ]
        >>> ontology = generate_ontology_llm_bottomup(triples, strategy='semantic', verbatim=True)
    """
    generator = LLMBasedBottomUpGenerator(generate_fn=generate_fn, verbatim=verbatim)
    return generator.generate_ontology(
        triples, 
        strategy=strategy,
        sample_size=sample_size,
        verbatim=verbatim
    )


def compare_bottomup_methods(triples: List[Dict],
                             generate_fn=None,
                             llm_strategies: List[str] = None,
                             verbatim: bool = False) -> Dict:
    """
    Compare different Bottom-Up ontology generation methods.
    
    Args:
        triples (List[Dict]): List of triples
        generate_fn: LLM generation function
        llm_strategies (List[str]): LLM strategies to compare - defaults to all ['instance', 'pattern', 'semantic']
        verbatim (bool): Print detailed logs
    
    Returns:
        Dict: Comparison results with all generated ontologies
    
    Example:
        >>> triples = [...]
        >>> results = compare_bottomup_methods(triples, verbatim=True)
        >>> print(f"Semantic method: {len(results['semantic']['entity_types'])} entities")
    """
    if llm_strategies is None:
        llm_strategies = ['instance', 'pattern', 'semantic']
    
    if verbatim:
        print("="*70)
        print(f"COMPARING BOTTOM-UP METHODS")
        print("="*70)
    
    results = {}
    
    # Rule-based method
    if verbatim:
        print("\n" + "="*70)
        print("METHOD: Rule-Based (Frequency)")
        print("="*70)
    
    results['rule_based'] = generate_ontology_from_triples(
        triples, method='frequency', verbatim=verbatim
    )
    
    # LLM-based methods
    for strategy in llm_strategies:
        if verbatim:
            print("\n" + "="*70)
            print(f"METHOD: LLM-Based ({strategy.capitalize()})")
            print("="*70)
        
        try:
            results[strategy] = generate_ontology_llm_bottomup(
                triples, 
                generate_fn=generate_fn,
                strategy=strategy,
                verbatim=verbatim
            )
        except Exception as e:
            if verbatim:
                print(f"⚠️  Strategy '{strategy}' failed: {str(e)[:100]}")
            results[strategy] = None
    
    # Comparison stats
    stats = {}
    for method_name in ['rule_based'] + llm_strategies:
        onto = results.get(method_name)
        if onto:
            stats[f"{method_name}_entities"] = len(onto.get('entity_types', {}))
            stats[f"{method_name}_relations"] = len(onto.get('relation_types', {}))
        else:
            stats[f"{method_name}_entities"] = 0
            stats[f"{method_name}_relations"] = 0
    
    # Summary
    if verbatim:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        for method in ['rule_based'] + llm_strategies:
            onto = results.get(method)
            if onto:
                print(f"\n{method.upper()}:")
                print(f"  Entity types: {len(onto.get('entity_types', {}))}")
                print(f"  Relation types: {len(onto.get('relation_types', {}))}")
                if onto.get('entity_types'):
                    entities = list(onto['entity_types'].keys())[:3]
                    print(f"  Sample entities: {', '.join(entities)}")
    
    results['comparison_stats'] = stats
    return results


def ontology_to_graphschema(ontology: Dict) -> 'GraphSchema':
    """
    Convert generated ontology to GraphSchema format.
    
    Args:
        ontology (Dict): Generated ontology
    
    Returns:
        GraphSchema: Schema object for validation
    """
    from graph_generation import GraphSchema
    
    entity_types = {}
    relation_types = {}
    
    # Convert entity types
    for name, entity_info in ontology.get("entity_types", {}).items():
        entity_types[name] = {
            "properties": entity_info.get("properties", [])
        }
    
    # Convert relation types
    for name, relation_info in ontology.get("relation_types", {}).items():
        relation_types[name] = {
            "domain": relation_info.get("domain", "Thing"),
            "range": relation_info.get("range", "Thing")
        }
    
    return GraphSchema(entity_types=entity_types, relation_types=relation_types)




def save_ontology(ontology: Dict, filepath: Union[str, Path], 
                 format: str = "json", **kwargs) -> None:
    """
    Convenience function to save ontology.
    
    Args:
        ontology (Dict): Ontology to save
        filepath (Union[str, Path]): Output file path
        format (str): Output format
        **kwargs: Additional arguments passed to OntologySerializer.save_ontology
    """
    OntologySerializer.save_ontology(ontology, filepath, format, **kwargs)


def load_ontology(filepath: Union[str, Path], format: str = None) -> Dict:
    """
    Convenience function to load ontology.
    
    Args:
        filepath (Union[str, Path]): Input file path
        format (str): Input format (auto-detected if None)
    
    Returns:
        Dict: Loaded ontology
    """
    return OntologySerializer.load_ontology(filepath, format)





__all__ = [
    "EntityTypeInferenceMethod",
    "InferredEntityType",
    "InferredRelationType",
    "TripleAnalyzer",
    "TopDownOntologyExtractor",
    "BottomUpOntologyInducer",
    "LLMBasedBottomUpGenerator",
    "OntologyMerger",
    "OntologySerializer",
    # Top-down methods
    "generate_ontology_from_questions",
    # Bottom-up methods
    "generate_ontology_from_triples",
    "generate_ontology_llm_bottomup",
    "compare_bottomup_methods",
    # Utilities
    "ontology_to_graphschema",
    "save_ontology",
    "load_ontology",
    # CQ-based generator classes
    "CQbyCQGenerator",
    "MemorylessCQbyCQGenerator",
    "OntogeniaGenerator",
    # Comparison utility
    "compare_cq_methods",
]
