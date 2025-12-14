"""
CQ-based Ontology Generation Methods Examples
==============================================

Demonstrates the three methods from the paper "Ontology Generation using Large Language Models":
1. CQbyCQ - Iterative processing with memory
2. Memoryless CQbyCQ - Independent processing with merging
3. Ontogenia - All-at-once processing

These methods automatically generate ontologies from competency questions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphConstruct import (
    # Individual generators
    CQbyCQGenerator,
    MemorylessCQbyCQGenerator,
    OntogeniaGenerator,
    # Convenience function
    generate_ontology_from_questions,
    compare_cq_methods,
    # Utilities
    save_ontology,
)

from Llms.llm_providers import get_generate_fn
import json


#############################################################
# Example 1: Basic usage of each method
#############################################################
def example_basic_usage():
    """Demonstrate basic usage of all three methods."""
    print("="*70)
    print("Example 1: Basic Usage of All Three Methods")
    print("="*70)
    
    # Sample competency questions
    questions = [
        "Which scientists work for which research institutions?",
        "What research projects are scientists involved in?",
        "Where are research institutions located?",
        "Which publications are authored by which scientists?",
    ]
    
    print(f"\nCompetency Questions ({len(questions)}):")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    
    # Initialize LLM
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Method 1: CQbyCQ (Iterative)
    print("\n" + "-"*70)
    print("Method 1: CQbyCQ (Iterative with Memory)")
    print("-"*70)
    
    ontology_cqbycq = generate_ontology_from_questions(
        questions, 
        generate_fn=generate,
        method='cqbycq',
        verbatim=True
    )
    
    print(f"\n✅ CQbyCQ Result:")
    print(f"   Entities: {len(ontology_cqbycq['entity_types'])}")
    print(f"   Relations: {len(ontology_cqbycq['relation_types'])}")
    
    # Method 2: Memoryless CQbyCQ
    print("\n" + "-"*70)
    print("Method 2: Memoryless CQbyCQ (Independent Processing)")
    print("-"*70)
    
    ontology_memoryless = generate_ontology_from_questions(
        questions,
        generate_fn=generate,
        method='memoryless',
        verbatim=True
    )
    
    print(f"\n✅ Memoryless Result:")
    print(f"   Entities: {len(ontology_memoryless['entity_types'])}")
    print(f"   Relations: {len(ontology_memoryless['relation_types'])}")
    
    # Method 3: Ontogenia (All-at-once)
    print("\n" + "-"*70)
    print("Method 3: Ontogenia (All-at-Once)")
    print("-"*70)
    
    ontology_ontogenia = generate_ontology_from_questions(
        questions,
        generate_fn=generate,
        method='ontogenia',
        verbatim=True
    )
    
    print(f"\n✅ Ontogenia Result:")
    print(f"   Entities: {len(ontology_ontogenia['entity_types'])}")
    print(f"   Relations: {len(ontology_ontogenia['relation_types'])}")


#############################################################
# Example 2: Using generator classes directly
#############################################################
def example_using_classes():
    """Use generator classes directly for more control."""
    print("\n" + "="*70)
    print("Example 2: Using Generator Classes Directly")
    print("="*70)
    
    questions = [
        "Who are the employees of each company?",
        "What products does each company produce?",
        "Which markets do companies operate in?",
    ]
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Create generator instances
    cqbycq_gen = CQbyCQGenerator(generate_fn=generate, verbatim=True)
    memoryless_gen = MemorylessCQbyCQGenerator(generate_fn=generate, verbatim=True)
    ontogenia_gen = OntogeniaGenerator(generate_fn=generate, verbatim=True)
    
    # Generate ontologies
    print("\n🔬 Running CQbyCQ...")
    onto1 = cqbycq_gen.generate_ontology(questions)
    
    print("\n🔬 Running Memoryless CQbyCQ...")
    onto2 = memoryless_gen.generate_ontology(questions)
    
    print("\n🔬 Running Ontogenia...")
    onto3 = ontogenia_gen.generate_ontology(questions)
    
    print("\n✅ All methods completed!")


#############################################################
# Example 3: Comparing all methods
#############################################################
def example_comparison():
    """Compare all three methods side-by-side."""
    print("\n" + "="*70)
    print("Example 3: Method Comparison")
    print("="*70)
    
    questions = [
        "What courses are taught by which professors?",
        "Which students are enrolled in which courses?",
        "What departments do professors belong to?",
        "Which buildings host which departments?",
    ]
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Compare all methods
    results = compare_cq_methods(
        questions,
        generate_fn=generate,
        verbatim=True
    )
    
    # Print detailed comparison
    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)
    
    print("\n📊 Statistics:")
    stats = results['comparison_stats']
    print(f"  CQbyCQ:      {stats['cqbycq_entities']} entities, {stats['cqbycq_relations']} relations")
    print(f"  Memoryless:  {stats['memoryless_entities']} entities, {stats['memoryless_relations']} relations")
    print(f"  Ontogenia:   {stats['ontogenia_entities']} entities, {stats['ontogenia_relations']} relations")
    
    # Show entities from each method
    print("\n📋 Entity Types:")
    for method_name, onto_key in [("CQbyCQ", "cqbycq"), ("Memoryless", "memoryless"), ("Ontogenia", "ontogenia")]:
        entities = list(results[onto_key]['entity_types'].keys())
        print(f"\n  {method_name}:")
        for entity in entities:
            props = results[onto_key]['entity_types'][entity].get('properties', [])
            print(f"    - {entity}: {props}")
    
    # Show relations from each method
    print("\n🔗 Relation Types:")
    for method_name, onto_key in [("CQbyCQ", "cqbycq"), ("Memoryless", "memoryless"), ("Ontogenia", "ontogenia")]:
        relations = results[onto_key]['relation_types']
        print(f"\n  {method_name}:")
        for rel_name, rel_info in relations.items():
            domain = rel_info.get('domain', '?')
            range_ = rel_info.get('range', '?')
            print(f"    - {rel_name}: {domain} → {range_}")


#############################################################
# Example 4: Domain-specific ontology (Medical)
#############################################################
def example_medical_domain():
    """Generate medical domain ontology."""
    print("\n" + "="*70)
    print("Example 4: Medical Domain Ontology")
    print("="*70)
    
    medical_questions = [
        "Which patients are diagnosed with which diseases?",
        "What treatments are prescribed for which diseases?",
        "Which doctors specialize in which medical areas?",
        "What medications are prescribed by which doctors?",
        "Which medical tests are performed for which diagnoses?",
    ]
    
    print("\nMedical Competency Questions:")
    for i, q in enumerate(medical_questions, 1):
        print(f"  {i}. {q}")
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Use Ontogenia for this example (fastest for demonstration)
    print("\n🔬 Generating medical ontology using Ontogenia...")
    
    medical_ontology = generate_ontology_from_questions(
        medical_questions,
        generate_fn=generate,
        method='ontogenia',
        verbatim=True
    )
    
    # Display results
    print("\n" + "="*70)
    print("MEDICAL ONTOLOGY")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Entity types: {len(medical_ontology['entity_types'])}")
    print(f"  Relation types: {len(medical_ontology['relation_types'])}")
    
    print("\n🏥 Entity Types:")
    for entity_name, entity_info in medical_ontology['entity_types'].items():
        print(f"  {entity_name}:")
        print(f"    Properties: {entity_info.get('properties', [])}")
        print(f"    Description: {entity_info.get('description', 'N/A')}")
    
    print("\n💊 Relation Types:")
    for rel_name, rel_info in medical_ontology['relation_types'].items():
        print(f"  {rel_name}: {rel_info.get('domain')} → {rel_info.get('range')}")


#############################################################
# Example 5: Saving and loading ontologies
#############################################################
def example_save_load():
    """Demonstrate saving and loading ontologies."""
    print("\n" + "="*70)
    print("Example 5: Saving and Loading Ontologies")
    print("="*70)
    
    questions = [
        "Which books are written by which authors?",
        "What genres do books belong to?",
        "Which publishers publish which books?",
    ]
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    # Generate ontology
    print("\n🔬 Generating book domain ontology...")
    ontology = generate_ontology_from_questions(
        questions, 
        generate_fn=generate,
        method='ontogenia'
    )
    
    # Save in different formats
    output_dir = Path("./output/ontologies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving ontology to {output_dir}...")
    
    # Save as JSON
    save_ontology(ontology, output_dir / "book_ontology.json", format="json")
    
    # Save as YAML
    save_ontology(ontology, output_dir / "book_ontology.yaml", format="yaml")
    
    # Save as OWL
    save_ontology(ontology, output_dir / "book_ontology.owl", format="owl")
    
    print("\n✅ Ontology saved in multiple formats!")
    
    # Load back
    from GraphConstruct import load_ontology
    
    loaded = load_ontology(output_dir / "book_ontology.json")
    print(f"\n📂 Loaded ontology: {len(loaded['entity_types'])} entities, "
          f"{len(loaded['relation_types'])} relations")


#############################################################
# Example 6: Method selection guide
#############################################################
def example_method_selection_guide():
    """Guide for choosing the right method."""
    print("\n" + "="*70)
    print("Example 6: Method Selection Guide")
    print("="*70)
    
    guide = """
    🎯 WHEN TO USE EACH METHOD:
    
    1️⃣  CQbyCQ (Iterative with Memory)
       ✅ Use when:
          - Questions build upon each other
          - Need rich context accumulation
          - Want iterative refinement
          - Have moderate number of CQs (< 20)
       
       ⚠️  Considerations:
          - Can accumulate errors
          - Slower (sequential processing)
          - May be biased by early decisions
       
       📊 Best for: Educational domains, historical knowledge
    
    2️⃣  Memoryless CQbyCQ (Independent + Merge)
       ✅ Use when:
          - Questions are independent
          - Want robustness to individual failures
          - Can parallelize processing
          - Have many CQs (> 20)
       
       ⚠️  Considerations:
          - May miss cross-CQ relationships
          - Requires sophisticated merging
          - Potential inconsistencies
       
       📊 Best for: Survey data, diverse question sources
    
    3️⃣  Ontogenia (All-at-Once)
       ✅ Use when:
          - Need global consistency
          - Want fast results (single LLM call)
          - CQs fit in context window
          - Have small to medium CQ set (< 15)
       
       ⚠️  Considerations:
          - Limited by context window
          - No iterative refinement
          - All-or-nothing approach
       
       📊 Best for: Well-defined domains, rapid prototyping
    
    💡 RECOMMENDATION:
       - Start with Ontogenia for quick exploration
       - Use Memoryless for large-scale applications
       - Use CQbyCQ for domains requiring deep reasoning
    """
    
    print(guide)


#############################################################
# Example 7: Hybrid approach
#############################################################
def example_hybrid_approach():
    """Combine methods for best results."""
    print("\n" + "="*70)
    print("Example 7: Hybrid Approach")
    print("="*70)
    
    questions = [
        "Which researchers collaborate on which projects?",
        "What funding sources support which projects?",
        "Which institutions host which research groups?",
        "What publications result from which projects?",
    ]
    
    generate = get_generate_fn("openai", config={"model": "gpt-4"})
    
    print("\n🔬 Step 1: Generate ontologies with all methods...")
    results = compare_cq_methods(questions, generate_fn=generate, verbatim=False)
    
    print("\n🔬 Step 2: Merge results from all three methods...")
    
    # Combine entity types from all methods
    combined_entities = {}
    for method in ['cqbycq', 'memoryless', 'ontogenia']:
        for entity_name, entity_info in results[method]['entity_types'].items():
            if entity_name not in combined_entities:
                combined_entities[entity_name] = entity_info
            else:
                # Merge properties
                existing_props = set(combined_entities[entity_name].get('properties', []))
                new_props = set(entity_info.get('properties', []))
                combined_entities[entity_name]['properties'] = list(existing_props | new_props)
    
    # Combine relation types
    combined_relations = {}
    for method in ['cqbycq', 'memoryless', 'ontogenia']:
        for rel_name, rel_info in results[method]['relation_types'].items():
            if rel_name not in combined_relations:
                combined_relations[rel_name] = rel_info
    
    hybrid_ontology = {
        "entity_types": combined_entities,
        "relation_types": combined_relations,
        "metadata": {
            "method": "Hybrid (CQbyCQ + Memoryless + Ontogenia)",
            "source_methods": ["cqbycq", "memoryless", "ontogenia"]
        }
    }
    
    print(f"\n✅ Hybrid Ontology:")
    print(f"   Entities: {len(hybrid_ontology['entity_types'])}")
    print(f"   Relations: {len(hybrid_ontology['relation_types'])}")
    print(f"\n   Entity types: {', '.join(list(hybrid_ontology['entity_types'].keys()))}")
    print(f"   Relation types: {', '.join(list(hybrid_ontology['relation_types'].keys()))}")


#############################################################
# Main execution
#############################################################
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CQ-BASED ONTOLOGY GENERATION METHODS")
    print("Based on: 'Ontology Generation using Large Language Models'")
    print("="*70)
    
    # Uncomment to run specific examples:
    
    # example_basic_usage()
    # example_using_classes()
    # example_comparison()
    # example_medical_domain()
    # example_save_load()
    example_method_selection_guide()
    # example_hybrid_approach()
    
    print("\n" + "="*70)
    print("✅ Examples completed!")
    print("="*70)
    print("\nTo run other examples, uncomment the function calls in the main block.")
