"""
GraphConstruct: Historical Knowledge Graph Construction
========================================================

A comprehensive package for generating knowledge graphs from historical texts and PDF documents.

Features:
- Document parsing (text, PDF)
- Entity extraction and linking
- Relationship extraction
- Knowledge graph construction
- Graph export (GraphML, RDF, JSON)
- Integration with spatial data (DGGS)

Example:
    >>> from GraphConstruct import HistoricalKGBuilder
    >>> builder = HistoricalKGBuilder(llm_provider='openai')
    >>> 
    >>> # Load documents
    >>> documents = builder.load_documents('history_docs/')
    >>> 
    >>> # Extract entities and relationships
    >>> entities, relationships = builder.extract_knowledge(documents)
    >>> 
    >>> # Build and export graph
    >>> graph = builder.build_graph(entities, relationships)
    >>> builder.export_graph(graph, 'output/history_kg.graphml')
"""

__version__ = "0.1.0"
__author__ = "MIT GraphReasoning Team"

from .document_processor import DocumentProcessor, Document
from .entity_extractor import EntityExtractor, Entity, EntityType
from .relationship_extractor import RelationshipExtractor, Relationship, RelationType
from .graph_builder import KnowledgeGraphBuilder, KnowledgeGraph, Node, Edge
from .graph_constructor import HistoricalKGBuilder

# Import ontology/schema generation (no external dependencies, load first)
from .onto_generation import (
    # Classes and enums
    EntityTypeInferenceMethod,
    InferredEntityType,
    InferredRelationType,
    TripleAnalyzer,
    TopDownOntologyExtractor,
    BottomUpOntologyInducer,
    LLMBasedBottomUpGenerator,
    OntologyMerger,
    OntologySerializer,
    # CQ-based generator classes
    CQbyCQGenerator,
    MemorylessCQbyCQGenerator,
    OntogeniaGenerator,
    # Convenience functions
    generate_ontology_from_questions,
    generate_ontology_from_triples,
    generate_ontology_llm_bottomup,
    compare_bottomup_methods,
    ontology_to_graphschema,
    save_ontology,
    load_ontology,
    # Comparison utility
    compare_cq_methods,
)

# Import graph generation functions and schema support (may have external dependencies)
try:
    from .graph_generation import (
        # Schema and validation
        GraphSchema,
        validate_and_filter_triples,
        normalize_entity_names,
        
        # Core graph generation functions
        extract,
        documents2Dataframe,
        concepts2Df,
        df2Graph,
        graph2Df,
        graphPrompt,
        colors2Community,
        contextual_proximity,
        make_graph_from_text,
        add_new_subgraph_from_text,
    )
except ImportError as e:
    # Graph generation requires transformers, skip if not available
    import warnings
    warnings.warn(f"Graph generation module not available: {e}")
    GraphSchema = None
    validate_and_filter_triples = None
    normalize_entity_names = None
    extract = None
    documents2Dataframe = None
    concepts2Df = None
    df2Graph = None
    graph2Df = None
    graphPrompt = None
    colors2Community = None
    contextual_proximity = None
    make_graph_from_text = None
    add_new_subgraph_from_text = None

# Import triple extraction module (direct chunk to triples extraction)
try:
    from .triple_extractor import (
        # Main class
        TripleExtractor,
        
        # Convenience functions
        extract_triples_from_chunk,
        extract_triples_from_chunks,
        extract_triples_to_dataframe,
        
        # I/O functions
        save_triples,
        load_triples,
    )
except ImportError as e:
    # Triple extractor requires same dependencies as graph generation
    import warnings
    warnings.warn(f"Triple extractor module not available: {e}")
    TripleExtractor = None
    extract_triples_from_chunk = None
    extract_triples_from_chunks = None
    extract_triples_to_dataframe = None
    save_triples = None
    load_triples = None

__all__ = [
    "DocumentProcessor",
    "Document",
    "EntityExtractor",
    "Entity",
    "EntityType",
    "RelationshipExtractor",
    "Relationship",
    "RelationType",
    "KnowledgeGraphBuilder",
    "KnowledgeGraph",
    "Node",
    "Edge",
    "HistoricalKGBuilder",
    # Schema and validation
    "GraphSchema",
    "validate_and_filter_triples",
    "normalize_entity_names",
    # Graph generation functions
    "extract",
    "documents2Dataframe",
    "concepts2Df",
    "df2Graph",
    "graph2Df",
    "graphPrompt",
    "colors2Community",
    "contextual_proximity",
    "make_graph_from_text",
    "add_new_subgraph_from_text",
    # Ontology/Schema generation
    "EntityTypeInferenceMethod",
    "InferredEntityType",
    "InferredRelationType",
    "TripleAnalyzer",
    "TopDownOntologyExtractor",
    "BottomUpOntologyInducer",
    "OntologyMerger",
    "OntologySerializer",
    "generate_ontology_from_questions",
    "generate_ontology_from_triples",
    "ontology_to_graphschema",
    "save_ontology",
    "load_ontology",
    # CQ-based ontology generation classes
    "CQbyCQGenerator",
    "MemorylessCQbyCQGenerator",
    "OntogeniaGenerator",
    # Comparison utility
    "compare_cq_methods",
    # Triple extraction (direct chunk to triples)
    "TripleExtractor",
    "extract_triples_from_chunk",
    "extract_triples_from_chunks",
    "extract_triples_to_dataframe",
    "save_triples",
    "load_triples",
]
