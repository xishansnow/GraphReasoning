"""
Historical Knowledge Graph Constructor
======================================

Main interface for building historical knowledge graphs from documents.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

from .document_processor import DocumentProcessor, Document
from .entity_extractor import EntityExtractor, Entity, EntityType
from .relationship_extractor import RelationshipExtractor, Relationship, RelationType
from .kg_builder import KnowledgeGraphBuilder, KnowledgeGraph


class HistoricalKGBuilder:
    """
    Main class for constructing historical knowledge graphs.
    
    Orchestrates the pipeline:
    1. Load documents (text, PDF, markdown)
    2. Extract entities (persons, places, events, dates)
    3. Extract relationships (temporal, spatial, social)
    4. Build knowledge graph
    5. Export in various formats
    """
    
    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None,
                 max_doc_size_mb: int = 100):
        """
        Initialize Historical KG Builder.
        
        Args:
            use_llm: Whether to use LLM for advanced extraction
            llm_provider: LLM provider ('openai', 'huggingface', etc.)
            max_doc_size_mb: Maximum document size in MB
        """
        self.document_processor = DocumentProcessor(max_size_mb=max_doc_size_mb)
        self.entity_extractor = EntityExtractor(use_llm=use_llm, llm_provider=llm_provider)
        self.relationship_extractor = RelationshipExtractor(use_llm=use_llm, llm_provider=llm_provider)
        self.kg_builder = KnowledgeGraphBuilder()
        
        self.documents: List[Document] = []
        self.entities: List[Entity] = []
        self.relationships: List[Relationship] = []
    
    def load_documents(self, source: Union[str, Path],
                      file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from directory or file.
        
        Args:
            source: Path to document(s)
            file_types: List of file extensions to load
        
        Returns:
            List of loaded documents
        """
        self.documents = self.document_processor.load_documents(source, file_types)
        print(f"✅ Loaded {len(self.documents)} documents")
        
        for doc in self.documents:
            print(f"   - {doc.metadata.get('filename', 'Unknown')}: "
                  f"{len(doc.content)} characters")
        
        return self.documents
    
    def extract_entities(self, min_confidence: float = 0.5,
                        entity_types: Optional[List[EntityType]] = None) -> List[Entity]:
        """
        Extract entities from loaded documents.
        
        Args:
            min_confidence: Minimum confidence threshold
            entity_types: Types of entities to extract
        
        Returns:
            List of extracted entities
        """
        self.entities = []
        
        for doc in self.documents:
            # Split document into chunks for better extraction
            chunks = self.document_processor.split_into_chunks(doc, chunk_size=500, overlap=50)
            
            for chunk in chunks:
                entities = self.entity_extractor.extract_entities(
                    chunk,
                    document_source=doc.metadata.get('filename'),
                    entity_types=entity_types
                )
                self.entities.extend(entities)
        
        # Deduplicate and filter
        self.entities = self.entity_extractor.deduplicate_entities(self.entities)
        self.entities = self.entity_extractor.filter_entities_by_confidence(
            self.entities, min_confidence
        )
        
        print(f"✅ Extracted {len(self.entities)} entities")
        
        # Print statistics
        type_counts = {}
        for entity in self.entities:
            entity_type = entity.entity_type.value
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        for entity_type, count in sorted(type_counts.items()):
            print(f"   - {entity_type}: {count}")
        
        return self.entities
    
    def extract_relationships(self, min_confidence: float = 0.5) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of extracted relationships
        """
        if not self.entities:
            print("⚠️  No entities to extract relationships from. "
                  "Run extract_entities() first.")
            return []
        
        self.relationships = []
        
        for doc in self.documents:
            relationships = self.relationship_extractor.extract_relationships(
                doc.content,
                entities=self.entities,
                document_source=doc.metadata.get('filename')
            )
            self.relationships.extend(relationships)
        
        # Deduplicate and filter
        self.relationships = self.relationship_extractor.deduplicate_relationships(
            self.relationships
        )
        self.relationships = self.relationship_extractor.filter_relationships_by_confidence(
            self.relationships, min_confidence
        )
        
        print(f"✅ Extracted {len(self.relationships)} relationships")
        
        # Print statistics
        rel_counts = {}
        for rel in self.relationships:
            rel_type = rel.predicate.value
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        for rel_type, count in sorted(rel_counts.items()):
            print(f"   - {rel_type}: {count}")
        
        return self.relationships
    
    def build_graph(self, name: str = "historical_kg") -> KnowledgeGraph:
        """
        Build knowledge graph from extracted entities and relationships.
        
        Args:
            name: Name for the knowledge graph
        
        Returns:
            KnowledgeGraph object
        """
        if not self.entities or not self.relationships:
            print("⚠️  Need both entities and relationships. "
                  "Run extract_entities() and extract_relationships() first.")
            return None
        
        graph = self.kg_builder.create_graph(
            name=name,
            entities=self.entities,
            relationships=self.relationships
        )
        
        stats = graph.get_statistics()
        print(f"✅ Built knowledge graph: {name}")
        print(f"   - Nodes: {stats['num_nodes']}")
        print(f"   - Edges: {stats['num_edges']}")
        
        return graph
    
    def export_graph(self, graph_name: str, output_path: str,
                    format: str = "json") -> None:
        """
        Export knowledge graph in various formats.
        
        Args:
            graph_name: Name of graph to export
            output_path: Output file path
            format: Export format ('json', 'graphml', 'csv', 'rdf')
        """
        if format == 'json':
            self.kg_builder.export_to_json(graph_name, output_path)
        elif format == 'graphml':
            self.kg_builder.export_to_graphml(graph_name, output_path)
        elif format == 'csv':
            self.kg_builder.export_to_csv(graph_name, output_path)
        elif format == 'rdf':
            self.kg_builder.export_to_rdf(graph_name, output_path)
        else:
            print(f"❌ Unknown format: {format}")
            print("Supported formats: json, graphml, csv, rdf")
    
    def run_pipeline(self, source: Union[str, Path],
                    output_dir: str = "output",
                    graph_name: str = "historical_kg",
                    min_confidence: float = 0.5) -> KnowledgeGraph:
        """
        Run complete pipeline: load -> extract entities -> extract relationships -> build graph.
        
        Args:
            source: Path to document(s)
            output_dir: Directory for output files
            graph_name: Name for knowledge graph
            min_confidence: Minimum confidence for entities/relationships
        
        Returns:
            Built KnowledgeGraph
        """
        print("\n" + "="*70)
        print("🌍 Historical Knowledge Graph Construction Pipeline")
        print("="*70)
        
        # Step 1: Load documents
        print("\n📚 Step 1: Loading documents...")
        self.load_documents(source)
        
        # Step 2: Extract entities
        print("\n🔍 Step 2: Extracting entities...")
        self.extract_entities(min_confidence=min_confidence)
        
        # Step 3: Extract relationships
        print("\n🔗 Step 3: Extracting relationships...")
        self.extract_relationships(min_confidence=min_confidence)
        
        # Step 4: Build graph
        print("\n🏗️  Step 4: Building knowledge graph...")
        graph = self.build_graph(name=graph_name)
        
        # Step 5: Export graph
        print("\n💾 Step 5: Exporting graph...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export in multiple formats
        self.export_graph(graph_name, str(output_path / f"{graph_name}.json"), format='json')
        self.export_graph(graph_name, str(output_path / f"{graph_name}.graphml"), format='graphml')
        self.export_graph(graph_name, str(output_path), format='csv')
        
        print("\n" + "="*70)
        print("✅ Pipeline completed successfully!")
        print("="*70)
        
        return graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        graph_stats = {}
        for name, graph in self.kg_builder.graphs.items():
            graph_stats[name] = graph.get_statistics()
        
        return {
            "documents_loaded": len(self.documents),
            "entities_extracted": len(self.entities),
            "relationships_extracted": len(self.relationships),
            "graphs": graph_stats
        }
