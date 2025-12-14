"""
GraphTools: Graph Analysis and Visualization Tools
===================================================

A comprehensive package for graph analysis, embeddings, visualization, and utilities.

Features:
- Node embedding generation and management
- Graph visualization (2D embeddings, clusters, density plots)
- Network analysis and statistics
- Graph simplification and optimization
- Community detection (Louvain algorithm)
- Graph persistence (save/load)

Example:
    >>> from GraphTools import generate_node_embeddings, visualize_embeddings_2d
    >>> from GraphTools import analyze_network, graph_Louvain
    >>> 
    >>> # Generate embeddings
    >>> embeddings = generate_node_embeddings(graph, tokenizer, model)
    >>> 
    >>> # Visualize
    >>> visualize_embeddings_2d(embeddings, data_dir='./output')
    >>> 
    >>> # Analyze network
    >>> stats = analyze_network(graph, data_dir='./analysis')
"""

__version__ = "0.1.0"
__author__ = "MIT GraphReasoning Team"

# Import all functions from graph_tools module
from .graph_tools import *

__all__ = [
    # Embedding functions
    "generate_node_embeddings",
    "save_embeddings",
    "load_embeddings",
    "find_best_fitting_node",
    "find_best_fitting_node_list",
    "update_node_embeddings",
    "regenerate_node_embeddings",
    
    # Visualization functions
    "visualize_embeddings_2d",
    "visualize_embeddings_2d_notext",
    "visualize_embeddings_2d_pretty",
    "visualize_embeddings_2d_pretty_and_sample",
    "visualize_embeddings_with_gmm_density_voronoi_and_print_top_samples",
    
    # Analysis functions
    "analyze_network",
    "graph_statistics_and_plots",
    "graph_statistics_and_plots_for_large_graphs",
    
    # Graph utilities
    "colors2Community",
    "graph_Louvain",
    "save_graph",
    "remove_small_fragents",
    "simplify_node_name_with_llm",
    "simplify_graph_simple",
    "simplify_graph",
    "simplify_graph_with_text",
    "make_HTML",
    "return_giant_component_of_graph",
    "return_giant_component_G_and_embeddings",
    "extract_number",
    "get_list_of_graphs_and_chunks",
    "print_graph_nodes_with_texts",
    "print_graph_nodes",
    "get_text_associated_with_node",
    "save_graph_with_text_as_JSON",
    "load_graph_with_text_as_JSON",
    "save_graph_without_text",
    "print_nodes_and_labels",
    "make_graph_from_text_withtext",
]
