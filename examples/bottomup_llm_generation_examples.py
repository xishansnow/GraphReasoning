"""
LLM-Based Bottom-Up Ontology Generation Examples
================================================

Demonstrates the paper "Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation"
using three extraction strategies:
1. Instance-level: Extract types from individual instances
2. Pattern-based: Identify patterns across multiple instances
3. Semantic: Deep semantic understanding using LLM

These methods generate ontologies from existing knowledge graph triples (bottom-up).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphConstruct import (
    # Generators
    LLMBasedBottomUpGenerator,
    # Convenience functions
    generate_ontology_llm_bottomup,
    compare_bottomup_methods,
    # Utilities
    save_ontology,
)

from Llms.llm_providers import get_generate_fn
import json


#############################################################
# Example 1: Basic usage of each strategy
#############################################################
def example_basic_usage():
    """Demonstrate basic usage of all three LLM strategies."""
    print("="*70)
    print("Example 1: Basic Usage of LLM-Based Bottom-Up Methods")
    print("="*70)
    
    # Sample triples (knowledge graph)
    triples = [
        {
            "node_1": "Alice",
            "node_2": "Bob",
            "edge": "knows",
            "node_1_type": "Person",
            "node_2_type": "Person"
        },
        {
            "node_1": "Bob",
            "node_2": "MIT",
            "edge": "works_for",
            "node_1_type": "Person",
            "node_2_type": "Organization"
        },
        {
            "node_1": "Alice",
            "node_2": "AI Research",
            "edge": "works_on",
            "node_1_type": "Person",
            "node_2_type": "Project"
        },
        {
            "node_1": "MIT",
            "node_2": "Cambridge",
            "edge": "located_in",
            "node_1_type": "Organization",
            "node_2_type": "Location"
        },
        {
            "node_1": "AI Research",
            "node_2": "MIT",
            "edge": "hosted_by",
            "node_1_type": "Project",
            "node_2_type": "Organization"
        },
    ]
    
    print(f"\nInput: {len(triples)} triples")
    for i, t in enumerate(triples, 1):
        print(f"  {i}. {t['node_1']} --{t['edge']}--> {t['node_2']}")
    
    # Initialize LLM
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Strategy 1: Instance-level
    print("\n" + "-"*70)
    print("Strategy 1: Instance-Level Extraction")
    print("-"*70)
    print("Extracts types from individual instances")
    
    ontology_instance = generate_ontology_llm_bottomup(
        triples,
        generate_fn=generate,
        strategy='instance',
        verbatim=True
    )
    
    print(f"\n✅ Result:")
    print(f"   Entities: {len(ontology_instance['entity_types'])}")
    print(f"   Relations: {len(ontology_instance['relation_types'])}")
    
    # Strategy 2: Pattern-based
    print("\n" + "-"*70)
    print("Strategy 2: Pattern-Based Extraction")
    print("-"*70)
    print("Identifies patterns across multiple instances")
    
    ontology_pattern = generate_ontology_llm_bottomup(
        triples,
        generate_fn=generate,
        strategy='pattern',
        verbatim=True
    )
    
    print(f"\n✅ Result:")
    print(f"   Entities: {len(ontology_pattern['entity_types'])}")
    print(f"   Relations: {len(ontology_pattern['relation_types'])}")
    
    # Strategy 3: Semantic
    print("\n" + "-"*70)
    print("Strategy 3: Semantic-Level Extraction")
    print("-"*70)
    print("Deep semantic understanding using LLM")
    
    ontology_semantic = generate_ontology_llm_bottomup(
        triples,
        generate_fn=generate,
        strategy='semantic',
        verbatim=True
    )
    
    print(f"\n✅ Result:")
    print(f"   Entities: {len(ontology_semantic['entity_types'])}")
    print(f"   Relations: {len(ontology_semantic['relation_types'])}")
    if ontology_semantic.get('metadata', {}).get('domain_topic'):
        print(f"   Domain/Topic: {ontology_semantic['metadata']['domain_topic']}")


#############################################################
# Example 2: Using generator class directly
#############################################################
def example_using_class():
    """Use LLMBasedBottomUpGenerator class directly for more control."""
    print("\n" + "="*70)
    print("Example 2: Using Generator Class Directly")
    print("="*70)
    
    # Academic publication triples
    triples = [
        {
            "node_1": "Deep Learning for NLP",
            "node_2": "John Smith",
            "edge": "authored_by",
            "node_1_type": "Paper",
            "node_2_type": "Researcher"
        },
        {
            "node_1": "John Smith",
            "node_2": "Stanford",
            "edge": "affiliated_with",
            "node_1_type": "Researcher",
            "node_2_type": "University"
        },
        {
            "node_1": "Deep Learning for NLP",
            "node_2": "NeurIPS 2023",
            "edge": "published_at",
            "node_1_type": "Paper",
            "node_2_type": "Conference"
        },
        {
            "node_1": "NeurIPS 2023",
            "node_2": "New Orleans",
            "edge": "held_at",
            "node_1_type": "Conference",
            "node_2_type": "Location"
        },
    ]
    
    print(f"\nDomain: Academic Publications")
    print(f"Triples: {len(triples)}")
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Create generator
    generator = LLMBasedBottomUpGenerator(
        generate_fn=generate,
        verbatim=True
    )
    
    # Generate with semantic strategy
    print("\n🔬 Generating ontology with semantic strategy...")
    ontology = generator.generate_ontology(
        triples,
        strategy='semantic',
        verbatim=True
    )
    
    # Display details
    print("\n" + "-"*70)
    print("Generated Ontology:")
    print("-"*70)
    
    print(f"\nEntity Types ({len(ontology['entity_types'])}):")
    for entity_name, entity_info in ontology['entity_types'].items():
        print(f"  {entity_name}:")
        if isinstance(entity_info, dict):
            print(f"    Definition: {entity_info.get('definition', 'N/A')}")
            print(f"    Properties: {entity_info.get('properties', [])}")
    
    print(f"\nRelation Types ({len(ontology['relation_types'])}):")
    for rel_name, rel_info in ontology['relation_types'].items():
        print(f"  {rel_name}:")
        if isinstance(rel_info, dict):
            print(f"    Domain → Range: {rel_info.get('domain', '?')} → {rel_info.get('range', '?')}")


#############################################################
# Example 3: Comparing all methods
#############################################################
def example_comparing_methods():
    """Compare rule-based and LLM-based methods."""
    print("\n" + "="*70)
    print("Example 3: Comparing Bottom-Up Methods")
    print("="*70)
    
    # Biomedical triples
    triples = [
        {
            "node_1": "Aspirin",
            "node_2": "Headache",
            "edge": "treats",
            "node_1_type": "Drug",
            "node_2_type": "Disease"
        },
        {
            "node_1": "Aspirin",
            "node_2": "Fever",
            "edge": "treats",
            "node_1_type": "Drug",
            "node_2_type": "Disease"
        },
        {
            "node_1": "Ibuprofen",
            "node_2": "Pain",
            "edge": "relieves",
            "node_1_type": "Drug",
            "node_2_type": "Symptom"
        },
        {
            "node_1": "Aspirin",
            "node_2": "Salicylic Acid",
            "edge": "contains",
            "node_1_type": "Drug",
            "node_2_type": "Chemical"
        },
        {
            "node_1": "Aspirin",
            "node_2": "Bayer",
            "edge": "manufactured_by",
            "node_1_type": "Drug",
            "node_2_type": "Company"
        },
    ]
    
    print(f"\nDomain: Biomedical")
    print(f"Triples: {len(triples)}")
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Compare methods
    print("\n🔬 Comparing Bottom-Up methods...")
    results = compare_bottomup_methods(
        triples,
        generate_fn=generate,
        llm_strategies=['instance', 'pattern', 'semantic'],
        verbatim=True
    )
    
    # Show stats
    print("\n" + "="*70)
    print("Comparison Results")
    print("="*70)
    
    stats = results['comparison_stats']
    for method in ['rule_based', 'instance', 'pattern', 'semantic']:
        entities = stats.get(f"{method}_entities", 0)
        relations = stats.get(f"{method}_relations", 0)
        print(f"\n{method.upper()}:")
        print(f"  Entities: {entities}")
        print(f"  Relations: {relations}")


#############################################################
# Example 4: Sampling large datasets
#############################################################
def example_sampling_large_dataset():
    """Handle large datasets by sampling."""
    print("\n" + "="*70)
    print("Example 4: Sampling Large Datasets")
    print("="*70)
    
    # Generate many triples (simulate large dataset)
    base_triples = [
        {
            "node_1": "Alice",
            "node_2": "Bob",
            "edge": "knows",
            "node_1_type": "Person",
            "node_2_type": "Person"
        },
        {
            "node_1": "Bob",
            "node_2": "MIT",
            "edge": "works_for",
            "node_1_type": "Person",
            "node_2_type": "Organization"
        },
        {
            "node_1": "MIT",
            "node_2": "Boston",
            "edge": "located_in",
            "node_1_type": "Organization",
            "node_2_type": "Location"
        },
    ]
    
    # Repeat to create larger dataset
    triples = base_triples * 10  # 30 triples
    
    print(f"\nDataset size: {len(triples)} triples")
    print("Using semantic strategy with sampling")
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Generate with sampling
    ontology = generate_ontology_llm_bottomup(
        triples,
        generate_fn=generate,
        strategy='semantic',
        sample_size=10,  # Sample 10 triples
        verbatim=True
    )
    
    print(f"\n✅ Generated ontology:")
    print(f"   Entities: {len(ontology['entity_types'])}")
    print(f"   Relations: {len(ontology['relation_types'])}")


#############################################################
# Example 5: Saving generated ontology
#############################################################
def example_saving_ontology():
    """Save generated ontology in various formats."""
    print("\n" + "="*70)
    print("Example 5: Saving Generated Ontology")
    print("="*70)
    
    triples = [
        {
            "node_1": "Alice",
            "node_2": "Bob",
            "edge": "knows",
            "node_1_type": "Person",
            "node_2_type": "Person"
        },
        {
            "node_1": "Bob",
            "node_2": "MIT",
            "edge": "works_for",
            "node_1_type": "Person",
            "node_2_type": "Organization"
        },
    ]
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    print("\n🔬 Generating ontology...")
    ontology = generate_ontology_llm_bottomup(
        triples,
        generate_fn=generate,
        strategy='semantic'
    )
    
    # Save in different formats
    output_dir = Path("./output/ontologies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_dir}...")
    
    # JSON format
    save_ontology(ontology, output_dir / "bottomup_ontology.json", format="json")
    
    # YAML format
    save_ontology(ontology, output_dir / "bottomup_ontology.yaml", format="yaml")
    
    print("\n✅ Ontology saved successfully!")


if __name__ == "__main__":
    # Uncomment to run examples
    
    # Basic usage (requires OpenAI API key)
    # example_basic_usage()
    
    # Using class directly (requires OpenAI API key)
    # example_using_class()
    
    # Comparing methods (requires OpenAI API key)
    # example_comparing_methods()
    
    # Sampling (requires OpenAI API key)
    # example_sampling_large_dataset()
    
    # Saving (requires OpenAI API key)
    # example_saving_ontology()
    
    print("📚 LLM-Based Bottom-Up Generation Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the three extraction strategies:")
    print("1. Instance-level: Extract types from individual instances")
    print("2. Pattern-based: Identify patterns across instances")
    print("3. Semantic: Deep semantic understanding")
    print("\nTo run examples, uncomment them and provide OpenAI API key.")
    print("\nKey functions:")
    print("  - generate_ontology_llm_bottomup(triples, strategy='semantic')")
    print("  - compare_bottomup_methods(triples)")
