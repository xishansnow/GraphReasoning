"""
DGGS Data to Knowledge Graph Conversion Module

Converts discretized geospatial data (CDL, SSURGO, etc.) into knowledge graph
triplets that can be integrated with the GraphReasoning framework.

This module provides:
1. Data discretization → RDF/Property Graph triplets
2. Spatial relationships (adjacency, containment, hierarchy)
3. Attribute enrichment and derivation
4. Graph structure generation for NetworkX/PyVis
5. Integration with existing knowledge graphs

Workflow:
  Discretized Data (DGGS cells with attributes)
       ↓
  Triplet Generation (RDF subject-predicate-object)
       ↓
  Spatial/Semantic Enrichment (relationships, hierarchies)
       ↓
  Graph Construction (NetworkX object)
       ↓
  Integration (merge with existing graphs)
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime


####################################################################
# Data Models
####################################################################
"""
Represents a spatial entity in the knowledge graph.

This class is a generic container for any spatial entity that needs to be represented
in the knowledge graph, not limited to DGGS cells. It can represent:

1. **DGGS Cells**: Discrete Global Grid System cells (e.g., 'cell_89c25a3')
2. **Administrative Regions**: Countries, states, counties (e.g., 'region_iowa')
3. **Crop Types**: Agricultural crops (e.g., 'crop_corn', 'crop_soybean')
4. **Soil Units**: Soil map units from SSURGO (e.g., 'soil_unit_12345')
5. **Farms/Fields**: Individual agricultural fields (e.g., 'farm_001')
6. **Land Use Types**: Different land use categories
7. **Any other spatial or thematic entity**: Custom domain objects

The class provides a unified interface to convert diverse spatial entities into
RDF-style triplets for knowledge graph construction, regardless of whether the
entity is discretized into DGGS cells or represented in other spatial frameworks.

Attributes:
    entity_id (str): Unique identifier for the entity
        Examples: 'cell_89c25a3', 'region_iowa', 'crop_corn', 'soil_unit_12345'
    
    entity_type (str): Classification of the entity
        Examples: 'DGGSCell', 'Region', 'Crop', 'SoilUnit', 'Farm', 'LandUse'
    
    attributes (Dict[str, Any]): Key-value pairs containing entity properties:
        For DGGS cells: resolution, area, geometry, parent_cell, child_cells
        For regions: population, area, boundary, administrative_level
        For crops: crop_name, growing_season, yield_estimate
        For soil: texture, drainage_class, pH, organic_matter
    
    created_at (datetime): Timestamp when the entity was instantiated

Methods:
    to_triplets() -> List[Tuple[str, str, str]]:
        Converts the entity to RDF-style subject-predicate-object triplets.
        
        Returns:
            List of tuples (subject, predicate, object) where:
                - subject: entity_id
                - predicate: relationship/property name with namespace prefix
                - object: value as string
        
        Example for DGGS cell:
            [
                ('cell_89c25a3', 'rdf:type', 'DGGSCell'),
                ('cell_89c25a3', 'attr:resolution', '9'),
                ('cell_89c25a3', 'attr:area', '125.67')
            ]
        
        Example for crop entity:
            [
                ('crop_corn', 'rdf:type', 'Crop'),
                ('crop_corn', 'attr:growing_season', 'summer'),
                ('crop_corn', 'attr:typical_yield', '180.5')
            ]
    
    to_dict() -> Dict[str, Any]:
        Converts the entity to a dictionary for serialization and storage.
        
        Returns:
            Dictionary with entity_id, entity_type, attributes, created_at

Example Usage:
    # DGGS cell entity
    >>> dggs_cell = SpatialEntity(
    ...     entity_id='cell_89c25a3',
    ...     entity_type='DGGSCell',
    ...     attributes={'resolution': 9, 'area': 125.67, 'is_land': True}
    ... )
    
    # Crop entity
    >>> crop = SpatialEntity(
    ...     entity_id='crop_corn',
    ...     entity_type='Crop',
    ...     attributes={'growing_season': 'summer', 'typical_yield': 180.5}
    ... )
    
    # Soil unit entity
    >>> soil = SpatialEntity(
    ...     entity_id='soil_unit_12345',
    ...     entity_type='SoilUnit',
    ...     attributes={'texture': 'clay_loam', 'drainage_class': 'moderate'}
    ... )
    
    # Convert all to triplets
    >>> dggs_triplets = dggs_cell.to_triplets()
    >>> crop_triplets = crop.to_triplets()
    >>> soil_triplets = soil.to_triplets()
"""
class SpatialEntity:
    
    """Represents a spatial entity in the knowledge graph.
    
    """
    
    def __init__(self, entity_id: str, entity_type: str, attributes: Dict[str, Any]):
        """
        Args:
            entity_id: Unique identifier (e.g., 'cell_89c25a3')
            entity_type: Type of entity (e.g., 'DGGSCell', 'Region', 'Crop')
            attributes: Dictionary of entity properties
        """
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.attributes = attributes
        self.created_at = datetime.now()
    
    def to_triplets(self) -> List[Tuple[str, str, str]]:
        """Convert entity to RDF-style triplets."""
        triplets = []
        
        # Type triplet
        triplets.append((self.entity_id, 'rdf:type', self.entity_type))
        
        # Attribute triplets
        for key, value in self.attributes.items():
            if value is not None:
                # Handle different value types
                if isinstance(value, bool):
                    value = 'true' if value else 'false'
                elif isinstance(value, (int, float)):
                    value = str(round(value, 2))
                else:
                    value = str(value)
                
                triplets.append((self.entity_id, f'attr:{key}', value))
        
        return triplets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'attributes': self.attributes,
            'created_at': self.created_at.isoformat()
        }


class SpatialRelationship:
    """Represents a relationship between spatial entities."""
    
    def __init__(self, source_id: str, target_id: str, rel_type: str, 
                 properties: Optional[Dict[str, Any]] = None):
        """
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Type of relationship (e.g., 'adjacent_to', 'contains')
            properties: Optional relationship properties
        """
        self.source_id = source_id
        self.target_id = target_id
        self.rel_type = rel_type
        self.properties = properties or {}
    
    def to_triplet(self) -> Tuple[str, str, str]:
        """Convert to RDF triplet."""
        return (self.source_id, self.rel_type, self.target_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'rel_type': self.rel_type,
            'properties': self.properties
        }


####################################################################
# Triplet Generation Functions
####################################################################

def discretized_agricultural_intensity_to_triplets(cell_token: str, intensity_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Convert agricultural intensity assessment to RDF triplets.
    
    Args:
        cell_token: DGGS cell token
        intensity_data: Output from discretize_cdl_agricultural_intensity()
    
    Returns:
        List of RDF triplets
    """
    triplets = []
    
    entity = SpatialEntity(
        f'ag_intensity_{cell_token}',
        'AgriculturalIntensity',
        {
            'intensity_level': intensity_data.get('intensity'),
            'intensity_score': intensity_data.get('intensity_score'),
            'is_monoculture': intensity_data.get('monoculture'),
            'agricultural_percent': intensity_data.get('ag_percent'),
        }
    )
    triplets.extend(entity.to_triplets())
    
    # Relationships
    intensity = intensity_data.get('intensity')
    if intensity:
        triplets.append((
            f'ag_intensity_{cell_token}',
            'has_intensity_category',
            f'intensity:{intensity}'
        ))
    
    return triplets


def spatial_adjacency_to_triplets(cell_tokens: List[str], grid) -> List[Tuple[str, str, str]]:
    """Generate spatial adjacency relationships between DGGS cells.
    
    Args:
        cell_tokens: List of DGGS cell tokens
        grid: DGGSS2 grid instance for computing neighbors
    
    Returns:
        List of adjacency triplets
    """
    triplets = []
    
    # Track processed pairs to avoid duplicates
    processed = set()
    
    for cell_token in cell_tokens:
        # Get neighbors for this cell
        neighbors = grid.cell_neighbors(cell_token)
        
        for neighbor_token in neighbors:
            # Create canonical pair to avoid reverse duplicates
            pair = tuple(sorted([cell_token, neighbor_token]))
            
            if pair not in processed:
                triplets.append((
                    f'cell_{cell_token}',
                    'adjacent_to',
                    f'cell_{neighbor_token}'
                ))
                processed.add(pair)
    
    return triplets


def temporal_triplets(cell_token: str, temporal_data: Dict[int, Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Generate temporal relationship triplets.
    
    Args:
        cell_token: DGGS cell token
        temporal_data: Data for multiple years from temporal analysis
    
    Returns:
        List of temporal triplets
    """
    triplets = []
    
    years = sorted(temporal_data.keys())
    
    for i, year in enumerate(years):
        year_data = temporal_data[year]
        
        # Year-specific entity
        triplets.append((
            f'cell_{cell_token}',
            f'state_in_{year}',
            f'state_{cell_token}_{year}'
        ))
        
        # Crop in that year
        crop = year_data.get('dominant_crop')
        if crop:
            triplets.append((
                f'state_{cell_token}_{year}',
                'dominant_crop',
                f'crop_{crop.lower().replace(" ", "_")}'
            ))
        
        # Temporal transitions
        if i > 0:
            prev_year = years[i-1]
            prev_crop = temporal_data[prev_year].get('dominant_crop')
            curr_crop = crop
            
            if prev_crop and curr_crop:
                triplets.append((
                    f'crop_{prev_crop.lower().replace(" ", "_")}',
                    f'transitions_to_in_{year}',
                    f'crop_{curr_crop.lower().replace(" ", "_")}'
                ))
    
    return triplets


####################################################################
# Graph Construction Functions
####################################################################

def create_knowledge_graph_from_discretized_data(
    discretized_data: Dict[str, Dict[str, Any]],
    triplet_converter = None,  # Function to convert cell data to triplets
    data_type: str = 'custom',  # Legacy parameter, use triplet_converter instead
    include_spatial: bool = True,
    grid = None
) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """Create NetworkX knowledge graph from discretized data.
    
    Args:
        discretized_data: Dictionary of discretized data {cell_token: data}
        triplet_converter: Function(cell_token, cell_data) -> List[Tuple] to convert data to triplets
                  Import from examples.raster_examples or examples.polygon_examples as needed
        data_type: (Deprecated) Use triplet_converter parameter instead
        include_spatial: Whether to include spatial adjacency relationships
        grid: DGGSS2 grid instance for spatial calculations
    
    Returns:
        Tuple of (NetworkX graph, list of triplet dictionaries)
        
    Example:
        from examples.raster_examples import discretized_cdl_to_triplets
        G, triplets = create_knowledge_graph_from_discretized_data(
            cdl_data, 
            triplet_converter=discretized_cdl_to_triplets
        )
    """
    G = nx.DiGraph()  # Directed graph for RDF representation
    triplets_list = []
    
    # Convert data to triplets
    for cell_token, cell_data in discretized_data.items():
        if triplet_converter:
            triplets = triplet_converter(cell_token, cell_data)
        elif data_type == 'intensity':
            # Keep backward compatibility for intensity (still in DGGS)
            triplets = discretized_agricultural_intensity_to_triplets(cell_token, cell_data)
        else:
            raise ValueError(
                f"Please provide a triplet_converter function. "
                f"Import from examples.raster_examples or examples.polygon_examples:\n"
                f"  from examples.raster_examples import discretized_cdl_to_triplets\n"
                f"  from examples.polygon_examples import discretized_ssurgo_to_triplets"
            )
        
        triplets_list.extend(triplets)
    
    # Add spatial relationships
    if include_spatial and grid:
        cell_tokens = list(discretized_data.keys())
        spatial_triplets = spatial_adjacency_to_triplets(cell_tokens, grid)
        triplets_list.extend(spatial_triplets)
    
    # Add triplets to graph
    for subject, predicate, obj in triplets_list:
        G.add_edge(subject, obj, relation=predicate, label=predicate)
        
        # Add node attributes
        if subject not in G.nodes:
            G.nodes[subject]['type'] = 'subject'
        if obj not in G.nodes:
            G.nodes[obj]['type'] = 'object'
    
    return G, [{'subject': s, 'predicate': p, 'object': o} for s, p, o in triplets_list]


def triplets_to_dataframe(triplets: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Convert triplet list to pandas DataFrame (for integration with GraphReasoning).
    
    Args:
        triplets: List of (subject, predicate, object) tuples
    
    Returns:
        DataFrame with columns 'node_1', 'edge', 'node_2'
    """
    df = pd.DataFrame(triplets, columns=['node_1', 'edge', 'node_2'])
    
    # Convert to strings first, then normalize node names
    df['node_1'] = df['node_1'].astype(str).str.lower()
    df['node_2'] = df['node_2'].astype(str).str.lower()
    
    return df


def merge_into_existing_graph(
    existing_graph: nx.Graph,
    new_triplets: List[Tuple[str, str, str]],
    merge_strategy: str = 'union'  # 'union', 'intersection', 'source_priority'
) -> nx.Graph:
    """Merge new spatial data triplets into existing knowledge graph.
    
    Args:
        existing_graph: Existing NetworkX graph
        new_triplets: New triplets to merge
        merge_strategy: How to handle conflicts
    
    Returns:
        Merged graph
    """
    G_new = nx.DiGraph()
    
    for subject, predicate, obj in new_triplets:
        G_new.add_edge(subject, obj, relation=predicate, label=predicate)
    
    if merge_strategy == 'union':
        G_merged = nx.compose(existing_graph, G_new)
    elif merge_strategy == 'intersection':
        # Only keep edges that exist in both graphs
        G_merged = existing_graph.copy()
        for u, v, data in G_new.edges(data=True):
            if G_merged.has_edge(u, v):
                # Keep existing, could update attributes
                pass
            else:
                G_merged.add_edge(u, v, **data)
    else:  # source_priority
        G_merged = G_new.copy()
        for u, v, data in existing_graph.edges(data=True):
            if not G_merged.has_edge(u, v):
                G_merged.add_edge(u, v, **data)
    
    return G_merged


####################################################################
# Export Functions
####################################################################

def export_triplets_to_csv(triplets: List[Tuple[str, str, str]], filepath: str):
    """Export triplets to CSV format (compatible with GraphReasoning).
    
    Args:
        triplets: List of triplets
        filepath: Output file path
    """
    df = triplets_to_dataframe(triplets)
    df.to_csv(filepath, sep='|', index=False, columns=['node_1', 'edge', 'node_2'])
    print(f"✅ Triplets exported to {filepath}")


def export_triplets_to_json(triplets: List[Dict[str, Any]], filepath: str):
    """Export triplets to JSON format.
    
    Args:
        triplets: List of triplet dictionaries
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(triplets, f, indent=2)
    print(f"✅ Triplets exported to {filepath}")


def export_graph_to_graphml(G: nx.Graph, filepath: str):
    """Export graph to GraphML format.
    
    Args:
        G: NetworkX graph
        filepath: Output file path
    """
    nx.write_graphml(G, filepath)
    print(f"✅ Graph exported to {filepath}")


def export_graph_to_rdf_turtle(triplets: List[Tuple[str, str, str]], filepath: str, 
                               namespace: str = 'http://example.com/geo/'):
    """Export triplets to RDF Turtle format.
    
    Args:
        triplets: List of triplets
        filepath: Output file path
        namespace: RDF namespace URI
    """
    content = f"@prefix geo: <{namespace}> .\n\n"
    
    for subject, predicate, obj in triplets:
        # Escape special characters
        subj_clean = subject.replace(':', '_').replace('/', '_')
        pred_clean = predicate.replace(':', '_').replace('/', '_')
        obj_clean = obj.replace(':', '_').replace('/', '_')
        
        if obj_clean.isdigit() or obj_clean.replace('.', '', 1).isdigit():
            # Numeric object
            content += f"geo:{subj_clean} geo:{pred_clean} {obj_clean} .\n"
        else:
            # String object
            content += f'geo:{subj_clean} geo:{pred_clean} "{obj_clean}" .\n'
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✅ RDF Turtle exported to {filepath}")


####################################################################
# Integration with GraphReasoning
####################################################################

def prepare_for_graph_reasoning(
    discretized_data: Dict[str, Dict[str, Any]],
    triplet_converter,  # Function to convert cell data to triplets
    data_type: str = 'custom',
    output_dir: str = './output/kg_output/'
) -> Tuple[List[Tuple[str, str, str]], nx.Graph]:
    """Prepare discretized data for integration with GraphReasoning framework.
    
    This function:
    1. Generates RDF triplets from discretized data
    2. Creates NetworkX graph
    3. Exports to multiple formats
    4. Returns formats suitable for graph_generation.make_graph_from_text()
    
    Args:
        discretized_data: Discretized DGGS cell data
        triplet_converter: Function(cell_token, cell_data) -> List[Tuple] to convert data
        data_type: Name for output files (default: 'custom')
        output_dir: Directory for output files
    
    Returns:
        Tuple of (triplets_list, networkx_graph)
        
    Example:
        from examples.raster_examples import discretized_cdl_to_triplets
        triplets, G = prepare_for_graph_reasoning(
            cdl_data,
            triplet_converter=discretized_cdl_to_triplets,
            data_type='cdl'
        )
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate triplets
    triplets = []
    for cell_token, cell_data in discretized_data.items():
        triplets.extend(triplet_converter(cell_token, cell_data))
    
    # Create graph
    G = nx.DiGraph()
    for subject, predicate, obj in triplets:
        G.add_edge(subject, obj, relation=predicate, label=predicate)
    
    # Export formats
    export_triplets_to_csv(triplets, str(output_path / f'{data_type}_triplets.csv'))
    export_graph_to_graphml(G, str(output_path / f'{data_type}_graph.graphml'))
    
    # Create triplet text format for graph_reasoning
    triplet_text = '\n'.join([
        f"{s}, {p}, {o}" for s, p, o in triplets
    ])
    
    with open(output_path / f'{data_type}_triplets.txt', 'w') as f:
        f.write(triplet_text)
    
    print(f"✅ Knowledge graph prepared in {output_dir}")
    print(f"   - Triplets: {len(triplets)}")
    print(f"   - Nodes: {G.number_of_nodes()}")
    print(f"   - Edges: {G.number_of_edges()}")
    
    return triplets, G
