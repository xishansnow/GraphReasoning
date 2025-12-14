"""
Ontology Generation Examples
============================

Demonstrates both top-down and bottom-up approaches to generate ontologies.
"""

import json
from typing import List, Dict

# Example 1: Top-Down Approach - From Competency Questions
# ========================================================

def example_top_down_extraction():
    """Extract ontology from competency questions."""
    from GraphConstruct import (
        TopDownOntologyExtractor,
        generate_ontology_from_questions
    )
    
    print("=" * 60)
    print("EXAMPLE 1: Top-Down Ontology Extraction")
    print("=" * 60)
    
    # Define competency questions
    competency_questions = [
        "Which scholars authored which publications?",
        "What research areas did scholars work in?",
        "Which institutions did scholars belong to?",
        "What conferences hosted which publications?",
        "Which publications cite which other publications?",
    ]
    
    print("\n📌 Competency Questions:")
    for i, q in enumerate(competency_questions, 1):
        print(f"   {i}. {q}")
    
    # Method 1: Using convenience function
    print("\n🔧 Method 1: Using convenience function...")
    ontology = generate_ontology_from_questions(
        competency_questions,
        verbatim=True
    )
    
    # Display results
    print("\n📊 Extracted Ontology:")
    print("\nEntity Types:")
    for entity_type, info in ontology["entity_types"].items():
        props = ", ".join(info['properties'])
        print(f"  • {entity_type}: [{props}]")
    
    print("\nRelation Types:")
    for relation_name, info in ontology["relation_types"].items():
        print(f"  • {relation_name}: {info['domain']} → {info['range']}")
    
    return ontology


# Example 2: Bottom-Up Approach - From Triples
# =============================================

def example_bottom_up_induction():
    """Induce ontology from triples data."""
    from GraphConstruct import (
        generate_ontology_from_triples,
        TripleAnalyzer
    )
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Bottom-Up Ontology Induction")
    print("=" * 60)
    
    # Sample triples data
    triples = [
        {
            "node_1": "Albert Einstein",
            "node_1_type": "Scholar",
            "edge": "worked_in",
            "node_2": "Physics",
            "node_2_type": "ResearchArea"
        },
        {
            "node_1": "Marie Curie",
            "node_1_type": "Scholar",
            "edge": "worked_in",
            "node_2": "Chemistry",
            "node_2_type": "ResearchArea"
        },
        {
            "node_1": "Albert Einstein",
            "node_1_type": "Scholar",
            "edge": "affiliated_with",
            "node_2": "Princeton University",
            "node_2_type": "Institution"
        },
        {
            "node_1": "Marie Curie",
            "node_1_type": "Scholar",
            "edge": "affiliated_with",
            "node_2": "University of Paris",
            "node_2_type": "Institution"
        },
        {
            "node_1": "Theory of Relativity",
            "node_1_type": "Publication",
            "edge": "authored_by",
            "node_2": "Albert Einstein",
            "node_2_type": "Scholar"
        },
        {
            "node_1": "Radioactivity Studies",
            "node_1_type": "Publication",
            "edge": "authored_by",
            "node_2": "Marie Curie",
            "node_2_type": "Scholar"
        },
        {
            "node_1": "Theory of Relativity",
            "node_1_type": "Publication",
            "edge": "cites",
            "node_2": "Classical Mechanics",
            "node_2_type": "Publication"
        },
    ]
    
    print(f"\n📊 Input: {len(triples)} triples")
    print("\nSample Triples:")
    for i, triple in enumerate(triples[:3], 1):
        print(f"   {i}. {triple['node_1']} --[{triple['edge']}]--> {triple['node_2']}")
    print(f"   ... and {len(triples) - 3} more")
    
    # Analyze triples
    print("\n🔍 Analyzing triples...")
    analyzer = TripleAnalyzer()
    analysis = analyzer.analyze_triples(triples)
    
    print(f"\n   • Total triples: {analysis['total_triples']}")
    print(f"   • Unique entities: {analysis['unique_entities_count']}")
    print(f"   • Unique relations: {analysis['unique_relations_count']}")
    
    # Infer ontology
    print("\n📈 Inferring ontology...")
    ontology = generate_ontology_from_triples(
        triples,
        min_frequency=1,
        verbatim=True
    )
    
    print("\n📊 Inferred Ontology:")
    print("\nEntity Types:")
    for entity_type, info in ontology['entity_types'].items():
        examples = ", ".join(info['examples'][:2])
        print(f"  • {entity_type} (freq: {info['frequency']}, ex: {examples})")
    
    print("\nRelation Types:")
    for relation_name, info in ontology['relation_types'].items():
        freq = info['frequency']
        print(f"  • {relation_name}: {info['domain']} → {info['range']} (freq: {freq})")
    
    return ontology, triples


# Example 3: Triple Analysis with Statistics
# ===========================================

def example_triple_analysis():
    """Detailed analysis of triples."""
    from GraphConstruct import TripleAnalyzer
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Detailed Triple Analysis")
    print("=" * 60)
    
    # More complex triples
    triples = [
        {"node_1": "John Smith", "node_1_type": "Person", "edge": "works_for", "node_2": "Microsoft", "node_2_type": "Organization"},
        {"node_1": "Jane Doe", "node_1_type": "Person", "edge": "works_for", "node_2": "Google", "node_2_type": "Organization"},
        {"node_1": "Bob Johnson", "node_1_type": "Person", "edge": "works_for", "node_2": "Microsoft", "node_2_type": "Organization"},
        {"node_1": "Microsoft", "node_1_type": "Organization", "edge": "located_in", "node_2": "Seattle", "node_2_type": "Location"},
        {"node_1": "Google", "node_1_type": "Organization", "edge": "located_in", "node_2": "Mountain View", "node_2_type": "Location"},
        {"node_1": "John Smith", "node_1_type": "Person", "edge": "participated_in", "node_2": "Tech Conference 2024", "node_2_type": "Event"},
        {"node_1": "Jane Doe", "node_1_type": "Person", "edge": "participated_in", "node_2": "AI Summit", "node_2_type": "Event"},
    ]
    
    analyzer = TripleAnalyzer()
    
    # Analyze triples
    print(f"\n📊 Analyzing {len(triples)} triples...\n")
    analysis = analyzer.analyze_triples(triples)
    
    print("Distribution Statistics:")
    print(f"\n  Entity Distribution:")
    entity_dist = analysis['entity_distribution']
    for entity, count in sorted(entity_dist.items(), key=lambda x: -x[1])[:5]:
        print(f"    • {entity}: {count} occurrences")
    
    print(f"\n  Relation Distribution:")
    rel_dist = analysis['relation_distribution']
    for relation, count in rel_dist.items():
        print(f"    • {relation}: {count} occurrences")
    
    # Infer entity types
    print(f"\n  Inferred Entity Types:")
    entity_types = analyzer.infer_entity_types(triples, verbatim=False)
    for name, entity_info in entity_types.items():
        examples = ", ".join(entity_info.examples[:2])
        print(f"    • {name}: {entity_info.frequency} instances (ex: {examples})")
    
    # Infer relation types with domain/range
    print(f"\n  Inferred Relation Types:")
    relation_types = analyzer.infer_relation_types(triples, verbatim=False)
    for name, rel_info in relation_types.items():
        print(f"    • {name}:")
        print(f"        Domain: {rel_info.domain}")
        print(f"        Range: {rel_info.range}")
        print(f"        Frequency: {rel_info.frequency}")
        print(f"        Confidence: {rel_info.confidence:.2%}")
    
    return analysis, entity_types, relation_types


# Example 4: Ontology Merging
# ============================

def example_ontology_merging():
    """Merge ontologies from different sources."""
    from GraphConstruct import (
        generate_ontology_from_questions,
        generate_ontology_from_triples,
        OntologyMerger
    )
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Ontology Merging")
    print("=" * 60)
    
    # Top-down ontology
    print("\n📌 Generating top-down ontology from questions...")
    questions = [
        "Which scientists work at which institutions?",
        "What research areas do scientists study?",
        "Which publications are in which journals?",
    ]
    onto_topdown = generate_ontology_from_questions(questions)
    print(f"   ✓ Generated {len(onto_topdown['entity_types'])} entity types")
    print(f"   ✓ Generated {len(onto_topdown['relation_types'])} relation types")
    
    # Bottom-up ontology
    print("\n📌 Generating bottom-up ontology from triples...")
    triples = [
        {"node_1": "Dr. Smith", "node_1_type": "Researcher", "edge": "works_at", "node_2": "MIT", "node_2_type": "University"},
        {"node_1": "Dr. Jones", "node_1_type": "Researcher", "edge": "works_at", "node_2": "Stanford", "node_2_type": "University"},
        {"node_1": "Dr. Smith", "node_1_type": "Researcher", "edge": "studies", "node_2": "Machine Learning", "node_2_type": "Topic"},
    ]
    onto_bottomup = generate_ontology_from_triples(triples, min_frequency=1)
    print(f"   ✓ Generated {len(onto_bottomup['entity_types'])} entity types")
    print(f"   ✓ Generated {len(onto_bottomup['relation_types'])} relation types")
    
    # Merge - Union
    print("\n🔀 Merging ontologies (UNION strategy)...")
    merged_union = OntologyMerger.merge_ontologies(
        [onto_topdown, onto_bottomup],
        strategy="union"
    )
    print(f"   • Entity types: {len(merged_union['entity_types'])}")
    print(f"   • Relation types: {len(merged_union['relation_types'])}")
    
    print("\n   Union Entity Types:")
    for et in list(merged_union['entity_types'].keys())[:5]:
        print(f"     • {et}")
    
    print("\n   Union Relation Types:")
    for rt in list(merged_union['relation_types'].keys())[:5]:
        print(f"     • {rt}")
    
    # Merge - Intersection
    print("\n🔀 Merging ontologies (INTERSECTION strategy)...")
    merged_intersection = OntologyMerger.merge_ontologies(
        [onto_topdown, onto_bottomup],
        strategy="intersection"
    )
    print(f"   • Entity types: {len(merged_intersection['entity_types'])}")
    print(f"   • Relation types: {len(merged_intersection['relation_types'])}")
    
    if merged_intersection['entity_types']:
        print("\n   Common Entity Types:")
        for et in merged_intersection['entity_types'].keys():
            print(f"     • {et}")
    else:
        print("\n   (No common entity types found)")
    
    if merged_intersection['relation_types']:
        print("\n   Common Relation Types:")
        for rt in merged_intersection['relation_types'].keys():
            print(f"     • {rt}")
    else:
        print("\n   (No common relation types found)")
    
    return merged_union, merged_intersection


# Example 5: Convert Ontology to GraphSchema
# ===========================================

def example_ontology_to_schema():
    """Convert generated ontology to GraphSchema."""
    from GraphConstruct import (
        generate_ontology_from_questions,
        ontology_to_graphschema
    )
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Convert Ontology to GraphSchema")
    print("=" * 60)
    
    # Generate ontology
    questions = [
        "Which persons work for organizations?",
        "Which locations host events?",
    ]
    ontology = generate_ontology_from_questions(questions)
    
    print("\n🔄 Converting ontology to GraphSchema...\n")
    
    # Convert to schema
    schema = ontology_to_graphschema(ontology)
    
    print("✓ GraphSchema created!")
    print(f"   • Entity types: {len(schema.entity_types)}")
    print(f"   • Relation types: {len(schema.relation_types)}")
    
    print("\n📋 Schema Details:")
    print("\n   Entity Types:")
    for entity_type, config in schema.entity_types.items():
        props = config.get('properties', [])
        print(f"     • {entity_type}: {props}")
    
    print("\n   Relation Types:")
    for rel_type, config in schema.relation_types.items():
        domain = config.get('domain', 'Thing')
        range_ = config.get('range', 'Thing')
        print(f"     • {rel_type}: {domain} → {range_}")
    
    return schema


# Example 6: Save and Load Ontology
# =================================

def example_save_load_ontology():
    """Save and load ontologies as JSON."""
    from GraphConstruct import generate_ontology_from_questions
    
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Save and Load Ontology")
    print("=" * 60)
    
    # Generate ontology
    questions = [
        "Which authors wrote which books?",
        "In which genres are books classified?",
    ]
    ontology = generate_ontology_from_questions(questions)
    
    # Save to JSON
    output_file = "output/ontology_sample.json"
    print(f"\n💾 Saving ontology to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(ontology, f, indent=2, default=str)
    
    print("✓ Saved!")
    
    # Load from JSON
    print(f"\n📂 Loading ontology from {output_file}...")
    with open(output_file, 'r') as f:
        loaded_ontology = json.load(f)
    
    print("✓ Loaded!")
    print(f"   • Entity types: {len(loaded_ontology['entity_types'])}")
    print(f"   • Relation types: {len(loaded_ontology['relation_types'])}")
    
    return loaded_ontology


# Main execution
# ==============

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run all examples
    print("\n" + "🚀" * 30)
    print("ONTOLOGY GENERATION EXAMPLES")
    print("🚀" * 30 + "\n")
    
    # Example 1
    onto_topdown = example_top_down_extraction()
    
    # Example 2
    onto_bottomup, sample_triples = example_bottom_up_induction()
    
    # Example 3
    analysis, entity_types, relation_types = example_triple_analysis()
    
    # Example 4
    merged_union, merged_intersection = example_ontology_merging()
    
    # Example 5
    schema = example_ontology_to_schema()
    
    # Example 6
    loaded_onto = example_save_load_ontology()
    
    print("\n" + "✨" * 30)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("✨" * 30)
