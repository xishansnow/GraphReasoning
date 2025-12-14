"""
Test for onto_generation module
Tests core functionality without external dependencies
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_triple_analyzer():
    """Test TripleAnalyzer class."""
    from GraphConstruct.onto_generation import TripleAnalyzer
    
    print("=" * 60)
    print("TEST 1: TripleAnalyzer")
    print("=" * 60)
    
    analyzer = TripleAnalyzer()
    
    triples = [
        {"node_1": "Alice", "node_1_type": "Person", "edge": "works_for", 
         "node_2": "MIT", "node_2_type": "Organization"},
        {"node_1": "Bob", "node_1_type": "Person", "edge": "works_for", 
         "node_2": "Google", "node_2_type": "Organization"},
        {"node_1": "MIT", "node_1_type": "Organization", "edge": "located_in", 
         "node_2": "Boston", "node_2_type": "Location"},
    ]
    
    # Test analyze_triples
    print("\n1️⃣ Testing analyze_triples()...")
    analysis = analyzer.analyze_triples(triples)
    
    assert analysis['total_triples'] == 3, "Should have 3 triples"
    assert analysis['unique_entities_count'] == 5, "Should have 5 unique entities (Alice, Bob, MIT, Google, Boston)"
    assert analysis['unique_relations_count'] == 2, "Should have 2 unique relations"
    
    print(f"   ✓ Total triples: {analysis['total_triples']}")
    print(f"   ✓ Unique entities: {analysis['unique_entities_count']}")
    print(f"   ✓ Unique relations: {analysis['unique_relations_count']}")
    
    # Test infer_entity_types
    print("\n2️⃣ Testing infer_entity_types()...")
    entity_types = analyzer.infer_entity_types(triples, min_frequency=1, verbatim=False)
    
    assert "Person" in entity_types, "Should infer Person type"
    assert "Organization" in entity_types, "Should infer Organization type"
    assert "Location" in entity_types, "Should infer Location type"
    
    print(f"   ✓ Inferred {len(entity_types)} entity types")
    for name in entity_types:
        print(f"     - {name}")
    
    # Test infer_relation_types
    print("\n3️⃣ Testing infer_relation_types()...")
    relation_types = analyzer.infer_relation_types(triples, min_frequency=1, verbatim=False)
    
    assert "works_for" in relation_types, "Should infer works_for relation"
    assert "located_in" in relation_types, "Should infer located_in relation"
    
    print(f"   ✓ Inferred {len(relation_types)} relation types")
    for name, info in relation_types.items():
        print(f"     - {name}: {info.domain} → {info.range}")
    
    print("\n✅ TripleAnalyzer tests passed!\n")


def test_top_down_extractor():
    """Test TopDownOntologyExtractor class."""
    from GraphConstruct.onto_generation import TopDownOntologyExtractor
    
    print("=" * 60)
    print("TEST 2: TopDownOntologyExtractor")
    print("=" * 60)
    
    extractor = TopDownOntologyExtractor()
    
    questions = [
        "Which persons work for organizations?",
        "What events did persons participate in?",
        "Which locations host events?"
    ]
    
    print("\n1️⃣ Testing extract_from_competency_questions()...")
    print(f"   Input questions: {len(questions)}")
    for q in questions:
        print(f"     - {q}")
    
    ontology = extractor.extract_from_competency_questions(questions, verbatim=False)
    
    assert "entity_types" in ontology, "Should have entity_types"
    assert "relation_types" in ontology, "Should have relation_types"
    
    print(f"\n   ✓ Extracted {len(ontology['entity_types'])} entity types")
    for name, info in ontology['entity_types'].items():
        print(f"     - {name}: {info['properties']}")
    
    print(f"\n   ✓ Extracted {len(ontology['relation_types'])} relation types")
    for name, info in ontology['relation_types'].items():
        print(f"     - {name}: {info['domain']} → {info['range']}")
    
    print("\n✅ TopDownOntologyExtractor tests passed!\n")


def test_bottom_up_inducer():
    """Test BottomUpOntologyInducer class."""
    from GraphConstruct.onto_generation import BottomUpOntologyInducer
    
    print("=" * 60)
    print("TEST 3: BottomUpOntologyInducer")
    print("=" * 60)
    
    inducer = BottomUpOntologyInducer()
    
    triples = [
        {"node_1": "Dr. Smith", "node_1_type": "Scientist", "edge": "works_at", 
         "node_2": "MIT", "node_2_type": "University"},
        {"node_1": "Dr. Jones", "node_1_type": "Scientist", "edge": "works_at", 
         "node_2": "Stanford", "node_2_type": "University"},
        {"node_1": "MIT", "node_1_type": "University", "edge": "located_in", 
         "node_2": "Boston", "node_2_type": "City"},
        {"node_1": "Dr. Smith", "node_1_type": "Scientist", "edge": "studies", 
         "node_2": "AI", "node_2_type": "ResearchArea"},
    ]
    
    print("\n1️⃣ Testing induce_ontology_from_triples()...")
    print(f"   Input triples: {len(triples)}")
    
    ontology = inducer.induce_ontology_from_triples(triples, min_frequency=1, verbatim=False)
    
    assert "entity_types" in ontology, "Should have entity_types"
    assert "relation_types" in ontology, "Should have relation_types"
    assert "statistics" in ontology, "Should have statistics"
    
    print(f"\n   ✓ Statistics:")
    stats = ontology['statistics']
    print(f"     - Total entities: {stats['total_entities']}")
    print(f"     - Total relations: {stats['total_relations']}")
    print(f"     - Total triples: {stats['total_triples']}")
    
    print(f"\n   ✓ Induced {len(ontology['entity_types'])} entity types")
    for name, info in ontology['entity_types'].items():
        print(f"     - {name}: {info['frequency']} instances")
    
    print(f"\n   ✓ Induced {len(ontology['relation_types'])} relation types")
    for name, info in ontology['relation_types'].items():
        print(f"     - {name}: {info['frequency']} occurrences")
    
    print("\n✅ BottomUpOntologyInducer tests passed!\n")


def test_ontology_merger():
    """Test OntologyMerger class."""
    from GraphConstruct.onto_generation import (
        OntologyMerger,
        TopDownOntologyExtractor,
        BottomUpOntologyInducer
    )
    
    print("=" * 60)
    print("TEST 4: OntologyMerger")
    print("=" * 60)
    
    # Create two simple ontologies
    onto1 = {
        "entity_types": {"Person": {}, "Organization": {}},
        "relation_types": {"works_for": {}, "located_in": {}}
    }
    
    onto2 = {
        "entity_types": {"Person": {}, "Location": {}},
        "relation_types": {"works_for": {}, "lives_in": {}}
    }
    
    print("\n1️⃣ Testing merge_ontologies() with UNION strategy...")
    merged_union = OntologyMerger.merge_ontologies([onto1, onto2], strategy="union")
    
    assert len(merged_union['entity_types']) == 3, "Union should have 3 entity types"
    assert len(merged_union['relation_types']) == 3, "Union should have 3 relation types"
    
    print(f"   ✓ Union entity types: {len(merged_union['entity_types'])}")
    print(f"     {list(merged_union['entity_types'].keys())}")
    print(f"   ✓ Union relation types: {len(merged_union['relation_types'])}")
    print(f"     {list(merged_union['relation_types'].keys())}")
    
    print("\n2️⃣ Testing merge_ontologies() with INTERSECTION strategy...")
    merged_inter = OntologyMerger.merge_ontologies([onto1, onto2], strategy="intersection")
    
    assert len(merged_inter['entity_types']) == 1, "Intersection should have 1 entity type"
    assert len(merged_inter['relation_types']) == 1, "Intersection should have 1 relation type"
    assert "Person" in merged_inter['entity_types'], "Person should be in intersection"
    assert "works_for" in merged_inter['relation_types'], "works_for should be in intersection"
    
    print(f"   ✓ Intersection entity types: {len(merged_inter['entity_types'])}")
    print(f"     {list(merged_inter['entity_types'].keys())}")
    print(f"   ✓ Intersection relation types: {len(merged_inter['relation_types'])}")
    print(f"     {list(merged_inter['relation_types'].keys())}")
    
    print("\n✅ OntologyMerger tests passed!\n")


def test_convenience_functions():
    """Test convenience functions."""
    from GraphConstruct.onto_generation import (
        generate_ontology_from_questions,
        generate_ontology_from_triples
    )
    
    print("=" * 60)
    print("TEST 5: Convenience Functions")
    print("=" * 60)
    
    print("\n1️⃣ Testing generate_ontology_from_questions()...")
    questions = [
        "Which authors wrote which books?",
        "In which genres are books classified?"
    ]
    
    onto_q = generate_ontology_from_questions(questions, verbatim=False)
    assert "entity_types" in onto_q, "Should return ontology dict"
    assert "relation_types" in onto_q, "Should return ontology dict"
    
    print(f"   ✓ Generated ontology with {len(onto_q['entity_types'])} entity types")
    print(f"   ✓ Generated ontology with {len(onto_q['relation_types'])} relation types")
    
    print("\n2️⃣ Testing generate_ontology_from_triples()...")
    triples = [
        {"node_1": "Book1", "node_1_type": "Book", "edge": "written_by", 
         "node_2": "Author1", "node_2_type": "Author"},
        {"node_1": "Book1", "node_1_type": "Book", "edge": "is_in_genre", 
         "node_2": "SciFi", "node_2_type": "Genre"},
    ]
    
    onto_t = generate_ontology_from_triples(triples, min_frequency=1, verbatim=False)
    assert "entity_types" in onto_t, "Should return ontology dict"
    assert "relation_types" in onto_t, "Should return ontology dict"
    assert "statistics" in onto_t, "Should include statistics"
    
    print(f"   ✓ Generated ontology with {len(onto_t['entity_types'])} entity types")
    print(f"   ✓ Generated ontology with {len(onto_t['relation_types'])} relation types")
    print(f"   ✓ Statistics: {onto_t['statistics']['total_triples']} triples")
    
    print("\n✅ Convenience functions tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "🚀" * 30)
    print("ONTO_GENERATION MODULE TESTS")
    print("🚀" * 30 + "\n")
    
    try:
        test_triple_analyzer()
        test_top_down_extractor()
        test_bottom_up_inducer()
        test_ontology_merger()
        test_convenience_functions()
        
        print("=" * 60)
        print("✨ ALL TESTS PASSED! ✨")
        print("=" * 60)
        print("\n✅ onto_generation module is fully functional!")
        print("📚 See ONTO_GENERATION_GUIDE.md for detailed documentation")
        print("📖 See ONTO_GENERATION_QUICK_REFERENCE.md for quick start")
        print("📝 See examples/onto_generation_examples.py for full examples\n")
        
        return True
    
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
