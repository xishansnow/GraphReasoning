"""
Triple Extractor Examples
==========================

Demonstrates how to use the TripleExtractor module to extract knowledge graph
triples directly from text chunks.

The triple_extractor module provides a streamlined interface for extracting
(subject, predicate, object) triples without the full graph construction pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphConstruct import (
    TripleExtractor,
    extract_triples_from_chunk,
    extract_triples_from_chunks,
    extract_triples_to_dataframe,
    save_triples,
    load_triples,
    GraphSchema,
)

from Llms.llm_providers import get_generate_fn


#############################################################
# Example 1: Extract triples from a single chunk
#############################################################
def example_single_chunk():
    """Extract triples from a single text chunk."""
    print("="*60)
    print("Example 1: Extract triples from single chunk")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Sample text chunk
    chunk = """
    Albert Einstein was born in Ulm, Germany in 1879. He developed the theory 
    of relativity, which revolutionized modern physics. Einstein received the 
    Nobel Prize in Physics in 1921 for his work on the photoelectric effect.
    He later moved to the United States and worked at Princeton University.
    """
    
    # Extract triples using convenience function
    triples = extract_triples_from_chunk(
        chunk=chunk,
        generate=generate,
        normalize=True,  # Normalize entity names
        verbatim=True    # Print detailed logs
    )
    
    # Display results
    print(f"\n✅ Extracted {len(triples)} triples:")
    for i, triple in enumerate(triples, 1):
        print(f"{i}. ({triple['node_1']}) --[{triple['edge']}]--> ({triple['node_2']})")


#############################################################
# Example 2: Extract triples from multiple chunks
#############################################################
def example_multiple_chunks():
    """Extract triples from multiple text chunks."""
    print("\n" + "="*60)
    print("Example 2: Extract triples from multiple chunks")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Sample text chunks
    chunks = [
        "Isaac Newton discovered the law of universal gravitation in 1687.",
        "Marie Curie was the first woman to win a Nobel Prize in 1903.",
        "Charles Darwin published 'On the Origin of Species' in 1859.",
    ]
    
    # Extract triples with progress bar
    triples = extract_triples_from_chunks(
        chunks=chunks,
        generate=generate,
        show_progress=True,
        normalize=True
    )
    
    # Display results
    print(f"\n✅ Extracted {len(triples)} triples from {len(chunks)} chunks:")
    for triple in triples:
        print(f"  • ({triple['node_1']}) --[{triple['edge']}]--> ({triple['node_2']})")


#############################################################
# Example 3: Extract triples with schema validation
#############################################################
def example_with_schema():
    """Extract triples with schema validation."""
    print("\n" + "="*60)
    print("Example 3: Extract triples with schema validation")
    print("="*60)
    
    # Define custom schema
    schema = GraphSchema(
        entity_types={
            "Person": {"properties": ["name", "birthdate"]},
            "Organization": {"properties": ["name", "location"]},
            "Location": {"properties": ["name", "country"]},
            "Achievement": {"properties": ["name", "year"]},
        },
        relation_types={
            "born_in": {"domain": "Person", "range": "Location"},
            "worked_at": {"domain": "Person", "range": "Organization"},
            "achieved": {"domain": "Person", "range": "Achievement"},
            "located_in": {"domain": "Organization", "range": "Location"},
        }
    )
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Sample text
    chunk = """
    Stephen Hawking was born in Oxford, England. He worked at the University 
    of Cambridge for most of his career. Hawking made groundbreaking 
    contributions to theoretical physics and cosmology.
    """
    
    # Create extractor with schema validation
    extractor = TripleExtractor(
        generate=generate,
        schema=schema,
        normalize_entities=True,
        validate_triples=True,  # Enable validation
        verbatim=True
    )
    
    # Extract triples
    triples = extractor.extract_from_chunk(chunk)
    
    # Display results
    print(f"\n✅ Valid triples after schema validation:")
    for triple in triples:
        print(f"  • ({triple['node_1']}) --[{triple['edge']}]--> ({triple['node_2']})")


#############################################################
# Example 4: Extract to DataFrame and save
#############################################################
def example_dataframe_and_save():
    """Extract triples to DataFrame and save to file."""
    print("\n" + "="*60)
    print("Example 4: Extract to DataFrame and save")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Sample chunks
    chunks = [
        "The Eiffel Tower is located in Paris, France.",
        "The Great Wall of China was built during the Ming Dynasty.",
        "The Colosseum in Rome was completed in 80 AD.",
    ]
    
    # Extract to DataFrame
    df = extract_triples_to_dataframe(
        chunks=chunks,
        generate=generate,
        normalize=True,
        show_progress=True
    )
    
    # Display DataFrame
    print(f"\n✅ Extracted triples DataFrame:")
    print(df[['node_1', 'edge', 'node_2']])
    
    # Save to file
    output_path = Path("./output/extracted_triples.csv")
    save_triples(df, output_path, format="csv")
    
    # Load back
    loaded_df = load_triples(output_path)
    print(f"\n✅ Loaded {len(loaded_df)} triples from file")


#############################################################
# Example 5: Batch processing with refinement
#############################################################
def example_batch_with_refinement():
    """Batch process chunks with iterative refinement."""
    print("\n" + "="*60)
    print("Example 5: Batch processing with refinement")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Sample historical text chunks
    chunks = [
        """
        The French Revolution began in 1789 with the storming of the Bastille.
        It led to the overthrow of the monarchy and the establishment of the
        First French Republic in 1792.
        """,
        """
        Napoleon Bonaparte rose to power during the French Revolution and
        became Emperor of France in 1804. He expanded French territory through
        military conquests across Europe.
        """,
    ]
    
    # Create extractor with refinement
    extractor = TripleExtractor(
        generate=generate,
        normalize_entities=True,
        validate_triples=False,
        verbatim=False
    )
    
    # Extract with 2 refinement iterations for better quality
    triples = extractor.extract_from_chunks(
        chunks=chunks,
        repeat_refine=2,  # More iterations = higher quality
        show_progress=True
    )
    
    # Display results
    print(f"\n✅ Extracted {len(triples)} triples with refinement:")
    for triple in triples[:10]:  # Show first 10
        print(f"  • ({triple['node_1']}) --[{triple['edge']}]--> ({triple['node_2']})")


#############################################################
# Example 6: Custom metadata attachment
#############################################################
def example_custom_metadata():
    """Attach custom metadata to extracted triples."""
    print("\n" + "="*60)
    print("Example 6: Attach custom metadata")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Sample chunks with metadata
    chunks = [
        "The Renaissance began in Italy during the 14th century.",
        "The Industrial Revolution started in Britain in the 18th century.",
    ]
    
    # Define per-chunk metadata
    metadata_list = [
        {"source": "history_textbook", "chapter": 5, "page": 42},
        {"source": "history_textbook", "chapter": 8, "page": 115},
    ]
    
    # Extract with metadata
    triples = extract_triples_from_chunks(
        chunks=chunks,
        generate=generate,
        metadata=metadata_list,  # Attach metadata
        show_progress=True
    )
    
    # Display results with metadata
    print(f"\n✅ Extracted triples with metadata:")
    for triple in triples:
        print(f"  • ({triple['node_1']}) --[{triple['edge']}]--> ({triple['node_2']})")
        print(f"    Source: {triple.get('source')}, Chapter: {triple.get('chapter')}, Page: {triple.get('page')}")


#############################################################
# Example 7: Compare with/without normalization
#############################################################
def example_normalization_comparison():
    """Compare results with and without entity normalization."""
    print("\n" + "="*60)
    print("Example 7: Normalization comparison")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    chunk = "Albert Einstein and einstein both worked on relativity theory."
    
    # Without normalization
    print("\n🔹 Without normalization:")
    triples_raw = extract_triples_from_chunk(
        chunk=chunk,
        generate=generate,
        normalize=False
    )
    for t in triples_raw:
        print(f"  • {t['node_1']} --[{t['edge']}]--> {t['node_2']}")
    
    # With normalization
    print("\n🔹 With normalization:")
    triples_normalized = extract_triples_from_chunk(
        chunk=chunk,
        generate=generate,
        normalize=True
    )
    for t in triples_normalized:
        print(f"  • {t['node_1']} --[{t['edge']}]--> {t['node_2']}")
    
    print(f"\n✅ Without normalization: {len(triples_raw)} triples")
    print(f"✅ With normalization: {len(triples_normalized)} triples (duplicates removed)")


#############################################################
# Example 8: Error handling and robustness
#############################################################
def example_error_handling():
    """Demonstrate error handling for problematic inputs."""
    print("\n" + "="*60)
    print("Example 8: Error handling")
    print("="*60)
    
    # Initialize LLM generator
    generate = get_generate_fn("openai", model="gpt-4")
    
    # Mix of good and problematic chunks
    chunks = [
        "Valid chunk: Einstein developed relativity.",
        "",  # Empty chunk
        "Another valid chunk: Newton discovered gravity.",
        "   ",  # Whitespace only
    ]
    
    # Extract (will handle errors gracefully)
    extractor = TripleExtractor(
        generate=generate,
        normalize_entities=True,
        verbatim=False
    )
    
    triples = []
    for i, chunk in enumerate(chunks):
        try:
            result = extractor.extract_from_chunk(chunk, chunk_id=f"chunk_{i}")
            if result:
                triples.extend(result)
                print(f"✅ Chunk {i}: Extracted {len(result)} triples")
            else:
                print(f"⚠️  Chunk {i}: No triples extracted")
        except Exception as e:
            print(f"❌ Chunk {i}: Error - {e}")
    
    print(f"\n✅ Total extracted: {len(triples)} triples from {len(chunks)} chunks")


#############################################################
# Main execution
#############################################################
if __name__ == "__main__":
    # Run all examples
    print("\n" + "="*60)
    print("TRIPLE EXTRACTOR EXAMPLES")
    print("="*60)
    
    # Uncomment to run specific examples:
    
    # example_single_chunk()
    # example_multiple_chunks()
    # example_with_schema()
    # example_dataframe_and_save()
    # example_batch_with_refinement()
    # example_custom_metadata()
    # example_normalization_comparison()
    # example_error_handling()
    
    print("\n✅ Examples completed!")
    print("\nTo run examples, uncomment the function calls in the main block.")
