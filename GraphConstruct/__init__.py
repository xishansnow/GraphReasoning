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
from .kg_builder import KnowledgeGraphBuilder, KnowledgeGraph, Node, Edge
from .graph_constructor import HistoricalKGBuilder

# Import graph generation functions
from .graph_generation import (
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
]
