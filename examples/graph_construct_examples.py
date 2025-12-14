"""
GraphConstruct Examples
=======================

Demonstrates how to use the GraphConstruct package to build historical knowledge graphs.
"""

from pathlib import Path
import sys

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from GraphConstruct import HistoricalKGBuilder


def example_1_simple_text_extraction():
    """Example 1: Extract knowledge graph from simple historical text."""
    print("\n" + "="*70)
    print("Example 1: Simple Text Extraction")
    print("="*70)
    
    # Sample historical text
    historical_text = """
    Thomas Jefferson, the third President of the United States, was born in 
    Shadwell, Virginia in 1743. He married Martha Wayles in 1772 in Virginia.
    Jefferson served as President from 1801 to 1809 and died in 1826.
    
    During his presidency, Jefferson oversaw the Louisiana Purchase in 1803,
    which doubled the size of the United States. The Purchase was negotiated
    with France and expanded American territory west.
    
    Benjamin Franklin, Jefferson's contemporary, lived in Philadelphia and 
    helped draft the Declaration of Independence in 1776. Franklin was born
    in Boston, Massachusetts in 1706.
    """
    
    # Save text to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(historical_text)
        temp_file = f.name
    
    try:
        # Create builder
        builder = HistoricalKGBuilder()
        
        # Load documents
        documents = builder.load_documents(temp_file)
        
        # Extract entities
        entities = builder.extract_entities(min_confidence=0.5)
        
        # Print extracted entities
        print("\n📌 Extracted Entities:")
        for entity in entities[:10]:  # Show first 10
            print(f"  - {entity.name} ({entity.entity_type.value}): "
                  f"confidence={entity.confidence:.2f}")
        
        # Extract relationships
        relationships = builder.extract_relationships(min_confidence=0.5)
        
        # Print extracted relationships
        print("\n🔗 Extracted Relationships:")
        for rel in relationships[:10]:  # Show first 10
            print(f"  - {rel.subject.name} --[{rel.predicate.value}]--> {rel.object.name}")
        
    finally:
        # Clean up
        import os
        os.unlink(temp_file)


def example_2_full_pipeline():
    """Example 2: Run full knowledge graph construction pipeline."""
    print("\n" + "="*70)
    print("Example 2: Full Pipeline")
    print("="*70)
    
    # Create sample documents
    import tempfile
    import os
    
    sample_docs = {
        "medieval_history.txt": """
        King Richard the Lionheart ruled England from 1189 to 1199.
        He led the Crusades against Saladin in the Holy Land.
        Richard was born in Oxford in 1157 and died in 1199.
        
        Queen Eleanor of Aquitaine was Richard's mother.
        She was born in 1122 in Aquitaine, France.
        Eleanor married King Henry II of England.
        """,
        
        "american_revolution.txt": """
        George Washington was the first President of the United States.
        Washington led the American Revolutionary War from 1775 to 1783.
        The war resulted in American independence from Great Britain.
        
        Thomas Jefferson wrote the Declaration of Independence in 1776.
        Benjamin Franklin and John Adams also signed the declaration.
        The declaration was signed in Philadelphia, Pennsylvania.
        """
    }
    
    # Create temporary directory with documents
    temp_dir = tempfile.mkdtemp()
    
    try:
        for filename, content in sample_docs.items():
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Run full pipeline
        builder = HistoricalKGBuilder()
        graph = builder.run_pipeline(
            source=temp_dir,
            output_dir="output/historical_kg",
            graph_name="history_kg",
            min_confidence=0.5
        )
        
        # Print statistics
        stats = builder.get_statistics()
        print("\n📊 Pipeline Statistics:")
        print(f"  - Documents: {stats['documents_loaded']}")
        print(f"  - Entities: {stats['entities_extracted']}")
        print(f"  - Relationships: {stats['relationships_extracted']}")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def example_3_entity_filtering():
    """Example 3: Extract specific entity types."""
    print("\n" + "="*70)
    print("Example 3: Entity Type Filtering")
    print("="*70)
    
    from GraphConstruct import EntityType
    
    historical_text = """
    The Battle of Hastings occurred on October 14, 1066 in Hastings, England.
    William the Conqueror defeated King Harold Godwinson.
    This battle resulted in the Norman Conquest of England.
    
    After the victory, William was crowned King of England on December 25, 1066.
    He established the Norman Dynasty which ruled England for centuries.
    """
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(historical_text)
        temp_file = f.name
    
    try:
        builder = HistoricalKGBuilder()
        documents = builder.load_documents(temp_file)
        
        # Extract only specific entity types
        print("\n🔍 Extracting PERSON and DATE entities only...")
        entities = builder.extract_entities(
            min_confidence=0.5,
            entity_types=[EntityType.PERSON, EntityType.DATE]
        )
        
        # Group by type
        by_type = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity)
        
        for entity_type, entity_list in sorted(by_type.items()):
            print(f"\n📌 {entity_type.upper()}:")
            for entity in entity_list:
                print(f"  - {entity.name}")
        
    finally:
        import os
        os.unlink(temp_file)


def example_4_graph_analysis():
    """Example 4: Analyze constructed knowledge graph."""
    print("\n" + "="*70)
    print("Example 4: Graph Analysis")
    print("="*70)
    
    historical_text = """
    Napoleon Bonaparte was a French military commander born in 1769 in Corsica.
    He became Emperor of France in 1804 and led the French Revolutionary Wars.
    
    Napoleon married Josephine in 1796 in Paris, France.
    Later, he divorced Josephine and married Marie Louise of Austria.
    
    Napoleon was defeated at the Battle of Waterloo in 1815 in Belgium.
    After his defeat, he was exiled to the island of Saint Helena.
    """
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(historical_text)
        temp_file = f.name
    
    try:
        builder = HistoricalKGBuilder()
        builder.load_documents(temp_file)
        builder.extract_entities(min_confidence=0.5)
        builder.extract_relationships(min_confidence=0.5)
        
        graph = builder.build_graph(name="napoleon_kg")
        
        # Get graph statistics
        stats = graph.get_statistics()
        
        print(f"\n📊 Graph Statistics:")
        print(f"  - Total nodes: {stats['num_nodes']}")
        print(f"  - Total edges: {stats['num_edges']}")
        
        print(f"\n📌 Node Types:")
        for node_type, count in sorted(stats['node_types'].items()):
            print(f"  - {node_type}: {count}")
        
        print(f"\n🔗 Edge Types:")
        for edge_type, count in sorted(stats['edge_types'].items()):
            print(f"  - {edge_type}: {count}")
        
        # Analyze specific nodes
        if graph.nodes:
            print(f"\n🔎 Sample Nodes (first 5):")
            for i, node in enumerate(list(graph.nodes.values())[:5]):
                neighbors = graph.get_neighbors(node.id)
                print(f"  - {node.label} ({node.entity_type}): {len(neighbors)} connections")
        
    finally:
        import os
        os.unlink(temp_file)


if __name__ == "__main__":
    import sys
    
    print("\n🌍 GraphConstruct Examples")
    print("="*70)
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == '1':
            example_1_simple_text_extraction()
        elif example_num == '2':
            example_2_full_pipeline()
        elif example_num == '3':
            example_3_entity_filtering()
        elif example_num == '4':
            example_4_graph_analysis()
        else:
            print(f"Unknown example: {example_num}")
            print("Available: 1, 2, 3, 4")
    else:
        # Run all examples
        example_1_simple_text_extraction()
        example_2_full_pipeline()
        example_3_entity_filtering()
        example_4_graph_analysis()
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)
        print("\n💡 Run specific examples:")
        print("  python examples/graph_construct_examples.py 1")
        print("  python examples/graph_construct_examples.py 2")
        print("  python examples/graph_construct_examples.py 3")
        print("  python examples/graph_construct_examples.py 4")
