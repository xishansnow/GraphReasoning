#!/usr/bin/env python3
"""
Verification script for function removal and migration.
Checks that removed functions are no longer available and replacements work correctly.
"""

import sys

print("=" * 70)
print("VERIFICATION: Function Removal and Migration")
print("=" * 70)

# Test 1: Verify removed functions are not available
print("\n[TEST 1] Verifying removed functions are no longer available...")
try:
    from GraphConstruct import generate_ontology_cqbycq
    print("❌ FAILED: generate_ontology_cqbycq should not be available")
    sys.exit(1)
except ImportError:
    print("✅ generate_ontology_cqbycq is correctly removed")

try:
    from GraphConstruct import generate_ontology_memoryless
    print("❌ FAILED: generate_ontology_memoryless should not be available")
    sys.exit(1)
except ImportError:
    print("✅ generate_ontology_memoryless is correctly removed")

try:
    from GraphConstruct import generate_ontology_ontogenia
    print("❌ FAILED: generate_ontology_ontogenia should not be available")
    sys.exit(1)
except ImportError:
    print("✅ generate_ontology_ontogenia is correctly removed")

# Test 2: Verify replacement function is available
print("\n[TEST 2] Verifying replacement function is available...")
try:
    from GraphConstruct import generate_ontology_from_questions
    print("✅ generate_ontology_from_questions is available")
except ImportError:
    print("❌ FAILED: generate_ontology_from_questions should be available")
    sys.exit(1)

# Test 3: Verify generator classes are still available
print("\n[TEST 3] Verifying generator classes are still available...")
try:
    from GraphConstruct import (
        CQbyCQGenerator,
        MemorylessCQbyCQGenerator,
        OntogeniaGenerator
    )
    print("✅ CQbyCQGenerator is available")
    print("✅ MemorylessCQbyCQGenerator is available")
    print("✅ OntogeniaGenerator is available")
except ImportError as e:
    print(f"❌ FAILED: Generator classes should be available: {e}")
    sys.exit(1)

# Test 4: Verify comparison tool is enhanced
print("\n[TEST 4] Verifying comparison tool is enhanced...")
try:
    from GraphConstruct import compare_cq_methods
    import inspect
    sig = inspect.signature(compare_cq_methods)
    if 'methods' in sig.parameters:
        print("✅ compare_cq_methods has 'methods' parameter")
    else:
        print("❌ FAILED: compare_cq_methods should have 'methods' parameter")
        sys.exit(1)
except ImportError:
    print("❌ FAILED: compare_cq_methods should be available")
    sys.exit(1)

# Test 5: Test generate_ontology_from_questions with method parameter
print("\n[TEST 5] Testing generate_ontology_from_questions with method parameter...")
try:
    from GraphConstruct import generate_ontology_from_questions
    
    questions = ["Who works where?", "Where are they located?"]
    
    # Test pattern method (no LLM needed)
    onto = generate_ontology_from_questions(questions, method='pattern')
    
    if 'entity_types' in onto and 'relation_types' in onto:
        print("✅ generate_ontology_from_questions works with method='pattern'")
    else:
        print("❌ FAILED: Unexpected ontology structure")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 6: Verify TopDownOntologyExtractor has method support
print("\n[TEST 6] Verifying TopDownOntologyExtractor method support...")
try:
    from GraphConstruct import TopDownOntologyExtractor
    import inspect
    
    sig = inspect.signature(TopDownOntologyExtractor.__init__)
    if 'method' in sig.parameters:
        print("✅ TopDownOntologyExtractor.__init__ has 'method' parameter")
    else:
        print("❌ FAILED: TopDownOntologyExtractor should have method parameter")
        sys.exit(1)
        
    # Test instantiation with method parameter
    extractor = TopDownOntologyExtractor(method='pattern')
    print("✅ TopDownOntologyExtractor can be instantiated with method parameter")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ ALL VERIFICATION TESTS PASSED")
print("=" * 70)
print("\nSummary:")
print("  ✓ Removed functions are no longer available")
print("  ✓ Replacement function (generate_ontology_from_questions) works")
print("  ✓ Generator classes are still available")
print("  ✓ Comparison tool is enhanced with methods parameter")
print("  ✓ TopDownOntologyExtractor has method support")
print("\nMigration complete!")
