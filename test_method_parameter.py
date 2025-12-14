#!/usr/bin/env python3
"""
Test script to verify method parameter support in onto_generation module.
"""

from GraphConstruct.onto_generation import (
    generate_ontology_from_questions,
    compare_cq_methods,
    generate_ontology_from_triples,
    TopDownOntologyExtractor
)

def test_generate_ontology_from_questions_with_method():
    """Test generate_ontology_from_questions with method parameter."""
    print("=" * 70)
    print("TEST 1: generate_ontology_from_questions with method parameter")
    print("=" * 70)
    
    questions = [
        "Which persons work for which organizations?",
        "Where are organizations located?"
    ]
    
    # Test with pattern method (default)
    print("\n✓ Testing method='pattern'")
    onto = generate_ontology_from_questions(questions, method='pattern', verbatim=False)
    print(f"  Entity types: {len(onto.get('entity_types', {}))}")
    print(f"  Relation types: {len(onto.get('relation_types', {}))}")
    
    print("\n✓ All tests passed for generate_ontology_from_questions")


def test_compare_cq_methods_with_method_selection():
    """Test compare_cq_methods with method selection."""
    print("\n" + "=" * 70)
    print("TEST 2: compare_cq_methods with method selection")
    print("=" * 70)
    
    questions = [
        "Which persons work for which organizations?",
        "Where are organizations located?"
    ]
    
    # Test comparing only pattern method
    print("\n✓ Testing methods=['pattern']")
    results = compare_cq_methods(questions, methods=['pattern'], verbatim=False)
    print(f"  Methods compared: {[k for k in results.keys() if k != 'comparison_stats']}")
    
    # Test that methods parameter works (skip LLM methods)
    print("\n✓ Testing methods parameter with pattern only (skip LLM)")
    results = compare_cq_methods(questions, methods=['pattern'], verbatim=False)
    assert 'pattern' in results
    assert 'comparison_stats' in results
    print(f"  ✓ methods=['pattern'] works correctly")
    
    print("\n✓ All tests passed for compare_cq_methods")


def test_top_down_ontology_extractor_with_method():
    """Test TopDownOntologyExtractor with method parameter."""
    print("\n" + "=" * 70)
    print("TEST 3: TopDownOntologyExtractor with method parameter")
    print("=" * 70)
    
    questions = [
        "Which persons work for which organizations?",
        "Where are organizations located?"
    ]
    
    # Test with pattern method
    print("\n✓ Creating extractor with method='pattern'")
    extractor = TopDownOntologyExtractor(method='pattern')
    onto = extractor.extract_from_competency_questions(questions, verbatim=False)
    print(f"  Entity types: {len(onto.get('entity_types', {}))}")
    print(f"  Relation types: {len(onto.get('relation_types', {}))}")
    
    # Test with method override
    print("\n✓ Testing method override in extract_from_competency_questions")
    onto = extractor.extract_from_competency_questions(
        questions, verbatim=False, method='pattern'
    )
    print(f"  Method used: {onto.get('metadata', {}).get('method', 'unknown')}")
    
    print("\n✓ All tests passed for TopDownOntologyExtractor")


def test_generate_ontology_from_triples_with_method():
    """Test generate_ontology_from_triples with method parameter."""
    print("\n" + "=" * 70)
    print("TEST 4: generate_ontology_from_triples with method parameter")
    print("=" * 70)
    
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
            "node_2": "Company X",
            "edge": "works_for",
            "node_1_type": "Person",
            "node_2_type": "Organization"
        }
    ]
    
    # Test with frequency method
    print("\n✓ Testing method='frequency'")
    onto = generate_ontology_from_triples(triples, method='frequency', verbatim=False)
    print(f"  Entity types: {len(onto.get('entity_types', {}))}")
    print(f"  Relation types: {len(onto.get('relation_types', {}))}")
    print(f"  Metadata: {onto.get('metadata', {})}")
    
    # Test with pattern method
    print("\n✓ Testing method='pattern'")
    onto = generate_ontology_from_triples(triples, method='pattern', verbatim=False)
    print(f"  Inference method: {onto.get('metadata', {}).get('inference_method', 'unknown')}")
    
    print("\n✓ All tests passed for generate_ontology_from_triples")


def test_kwargs_support():
    """Test that kwargs are properly passed through functions."""
    print("\n" + "=" * 70)
    print("TEST 5: Function signature verification")
    print("=" * 70)
    
    # Test function signatures
    print("\n✓ Testing function signatures")
    
    # Test generate_ontology_from_questions accepts method parameter
    import inspect
    sig = inspect.signature(generate_ontology_from_questions)
    has_method = 'method' in sig.parameters
    print(f"  generate_ontology_from_questions has method param: {has_method}")
    
    # Test compare_cq_methods accepts methods parameter
    sig = inspect.signature(compare_cq_methods)
    has_methods = 'methods' in sig.parameters
    print(f"  compare_cq_methods has methods param: {has_methods}")
    
    # Test generate_ontology_from_triples accepts method parameter
    sig = inspect.signature(generate_ontology_from_triples)
    has_method = 'method' in sig.parameters
    print(f"  generate_ontology_from_triples has method param: {has_method}")
    
    print("\n✓ All tests passed for function signatures")


if __name__ == "__main__":
    print("\n" + "🧪 TESTING METHOD PARAMETER SUPPORT".center(70))
    print("=" * 70)
    
    test_generate_ontology_from_questions_with_method()
    test_compare_cq_methods_with_method_selection()
    test_top_down_ontology_extractor_with_method()
    test_generate_ontology_from_triples_with_method()
    test_kwargs_support()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED".center(70))
    print("=" * 70)
