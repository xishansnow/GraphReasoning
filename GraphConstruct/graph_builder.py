"""
Knowledge Graph Builder Module
==============================

Constructs knowledge graphs from entities and relationships.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .entity_extractor import Entity, EntityType
from .relationship_extractor import Relationship, RelationType


@dataclass
class Node:
    """Graph node representing an entity."""
    
    id: str
    label: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.entity_type,
            "attributes": self.attributes
        }


@dataclass
class Edge:
    """Graph edge representing a relationship."""
    
    source: str
    target: str
    label: str
    relation_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "type": self.relation_type,
            "attributes": self.attributes
        }


class KnowledgeGraph:
    """Represents a knowledge graph."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize knowledge graph.
        
        Args:
            name: Graph name
            description: Graph description
        """
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.node_index: Dict[str, str] = {}  # Map entity names to node IDs
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.node_index[node.label] = node.id
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # Ensure source and target nodes exist
        if edge.source in self.nodes and edge.target in self.nodes:
            self.edges.append(edge)
    
    def get_node(self, entity_name: str) -> Optional[Node]:
        """Get node by entity name."""
        node_id = self.node_index.get(entity_name)
        return self.nodes.get(node_id) if node_id else None
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes."""
        neighbors = set()
        for edge in self.edges:
            if edge.source == node_id:
                neighbors.add(edge.target)
            elif edge.target == node_id:
                neighbors.add(edge.source)
        return list(neighbors)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types()
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in self.nodes.values():
            counts[node.entity_type] = counts.get(node.entity_type, 0) + 1
        return counts
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by type."""
        counts = {}
        for edge in self.edges:
            counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
        return counts


class KnowledgeGraphBuilder:
    """Builds knowledge graphs from entities and relationships."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graphs: Dict[str, KnowledgeGraph] = {}
    
    def create_graph(self, name: str, entities: List[Entity],
                    relationships: List[Relationship]) -> KnowledgeGraph:
        """
        Create a knowledge graph from entities and relationships.
        
        Args:
            name: Name for the graph
            entities: List of entities
            relationships: List of relationships
        
        Returns:
            KnowledgeGraph object
        """
        graph = KnowledgeGraph(name=name)
        
        # Add nodes from entities
        entity_to_node = {}
        for entity in entities:
            node_id = self._generate_node_id(entity.name)
            node = Node(
                id=node_id,
                label=entity.name,
                entity_type=entity.entity_type.value,
                attributes=entity.attributes or {}
            )
            graph.add_node(node)
            entity_to_node[entity.name] = node_id
        
        # Add edges from relationships
        for rel in relationships:
            source_id = entity_to_node.get(rel.subject.name)
            target_id = entity_to_node.get(rel.object.name)
            
            if source_id and target_id:
                edge = Edge(
                    source=source_id,
                    target=target_id,
                    label=rel.predicate.value,
                    relation_type=rel.predicate.value,
                    attributes={
                        'confidence': rel.confidence,
                        'context': rel.context
                    }
                )
                graph.add_edge(edge)
        
        self.graphs[name] = graph
        return graph
    
    def _generate_node_id(self, entity_name: str) -> str:
        """Generate unique node ID from entity name."""
        import hashlib
        name_hash = hashlib.md5(entity_name.encode()).hexdigest()[:8]
        return f"node_{name_hash}"
    
    def export_to_dict(self, graph_name: str) -> Dict[str, Any]:
        """
        Export graph to dictionary format.
        
        Args:
            graph_name: Name of graph to export
        
        Returns:
            Dictionary representation
        """
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        return {
            "name": graph.name,
            "description": graph.description,
            "created_at": graph.created_at.isoformat(),
            "statistics": graph.get_statistics(),
            "nodes": [node.to_dict() for node in graph.nodes.values()],
            "edges": [edge.to_dict() for edge in graph.edges]
        }
    
    def export_to_json(self, graph_name: str, output_path: str) -> None:
        """
        Export graph to JSON file.
        
        Args:
            graph_name: Name of graph to export
            output_path: Path to output JSON file
        """
        data = self.export_to_dict(graph_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Graph exported to {output_path}")
    
    def export_to_graphml(self, graph_name: str, output_path: str) -> None:
        """
        Export graph to GraphML format.
        
        Args:
            graph_name: Name of graph to export
            output_path: Path to output GraphML file
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not installed. Install with: pip install networkx")
            return
        
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph.nodes.values():
            G.add_node(node.id, **node.to_dict())
        
        # Add edges
        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, **edge.to_dict())
        
        # Export to GraphML
        nx.write_graphml(G, output_path)
        print(f"✅ Graph exported to {output_path}")
    
    def export_to_csv(self, graph_name: str, output_dir: str) -> None:
        """
        Export graph to CSV files (nodes and edges).
        
        Args:
            graph_name: Name of graph to export
            output_dir: Directory to save CSV files
        """
        import csv
        from pathlib import Path
        
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export nodes
        nodes_file = output_path / f"{graph_name}_nodes.csv"
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'label', 'type'])
            writer.writeheader()
            for node in graph.nodes.values():
                writer.writerow({
                    'id': node.id,
                    'label': node.label,
                    'type': node.entity_type
                })
        
        # Export edges
        edges_file = output_path / f"{graph_name}_edges.csv"
        with open(edges_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['source', 'target', 'label', 'type'])
            writer.writeheader()
            for edge in graph.edges:
                writer.writerow({
                    'source': edge.source,
                    'target': edge.target,
                    'label': edge.label,
                    'type': edge.relation_type
                })
        
        print(f"✅ Graph exported to {output_dir}")
        print(f"   - Nodes: {nodes_file}")
        print(f"   - Edges: {edges_file}")
    
    def export_to_rdf(self, graph_name: str, output_path: str,
                     base_uri: str = "http://example.org/history/") -> None:
        """
        Export graph to RDF Turtle format.
        
        Args:
            graph_name: Name of graph to export
            output_path: Path to output RDF file
            base_uri: Base URI for RDF entities
        """
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        rdf_content = []
        rdf_content.append("@prefix ex: <http://example.org/> .")
        rdf_content.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        rdf_content.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        rdf_content.append("")
        
        # Add nodes as RDF resources
        for node in graph.nodes.values():
            resource_uri = f"{base_uri}{node.id}"
            rdf_content.append(f"<{resource_uri}> rdf:type ex:{node.entity_type} ;")
            rdf_content.append(f'  rdfs:label "{node.label}" .')
            rdf_content.append("")
        
        # Add edges as RDF properties
        for edge in graph.edges:
            source_uri = f"{base_uri}{edge.source}"
            target_uri = f"{base_uri}{edge.target}"
            rdf_content.append(f"<{source_uri}> ex:{edge.relation_type} <{target_uri}> .")
            rdf_content.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(rdf_content))
        
        print(f"✅ Graph exported to RDF: {output_path}")
