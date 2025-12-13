"""DGGS S2 usage examples demonstrating hierarchical multi-scale relationships.

Based on "The S2 Hierarchical Discrete Global Grid as a Nexus for Data Representation, 
Integration, and Querying Across Geospatial Knowledge Graphs"
"""

from pathlib import Path
import sys

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from DGGS.dggs import DGGSS2, analyze_entity_relationships
import networkx as nx


def example_basic_cell_operations():
    """Basic S2 cell operations: conversion, center, neighbors."""
    print("=" * 60)
    print("Example 1: Basic Cell Operations")
    print("=" * 60)
    
    grid = DGGSS2(level=12)
    
    # Convert lat/lon to S2 cell token
    lat, lon = 42.3601, -71.0589  # Boston
    token = grid.latlon_to_token(lat, lon)
    print(f"Location ({lat}, {lon}) -> S2 token: {token}")
    
    # Get cell center
    center_lat, center_lon = grid.cell_center(token)
    print(f"Cell center: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Find neighbors
    neighbors = grid.neighbors(token, ring=1)
    print(f"Number of immediate neighbors: {len(neighbors)}")
    print(f"First 3 neighbors: {neighbors[:3]}")
    print()


def example_hierarchical_relationships():
    """Demonstrate parent-child relationships across different scales."""
    print("=" * 60)
    print("Example 2: Hierarchical (Cross-Scale) Relationships")
    print("=" * 60)
    
    grid = DGGSS2(level=12)
    
    # Start with a cell at level 12
    lat, lon = 42.3601, -71.0589
    token_l12 = grid.latlon_to_token(lat, lon, level=12)
    cell_l12 = grid.token_to_cell(token_l12)
    
    print(f"Original cell (level {cell_l12.level()}): {token_l12}")
    
    # Get parent at level 10 (coarser)
    parent_l10 = grid.parent(token_l12, parent_level=10)
    print(f"Parent cell (level 10): {parent_l10}")
    
    # Get parent at level 8 (even coarser)
    parent_l8 = grid.parent(token_l12, parent_level=8)
    print(f"Grandparent cell (level 8): {parent_l8}")
    
    # Get children at level 14 (finer)
    children_l14 = grid.children(token_l12, child_level=14)
    print(f"Number of children (level 14): {len(children_l14)}")
    print(f"First 3 children: {children_l14[:3]}")
    
    # Demonstrate hierarchical containment
    print("\nHierarchical containment:")
    print(f"  Level 8 (area ~1000 km²) contains")
    print(f"  Level 10 (area ~60 km²) contains")
    print(f"  Level 12 (area ~4 km²) contains")
    print(f"  Level 14 (area ~0.25 km²)")
    print()


def example_multi_scale_graph():
    """Build a multi-scale graph with parent-child and adjacency relationships."""
    print("=" * 60)
    print("Example 3: Multi-Scale Graph Construction")
    print("=" * 60)
    
    grid = DGGSS2(level=12)
    
    # Cover a small region around Boston
    cells = grid.cover_cap(42.3601, -71.0589, radius_km=5, level=12)
    print(f"Covering Boston area (~5km radius) with {len(cells)} level-12 cells")
    
    # Build hierarchical graph with parents
    H = grid.build_hierarchical_graph(cells, include_parents=True, include_children=False)
    
    print(f"\nGraph statistics:")
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")
    
    # Count edge types
    edge_types = {}
    for u, v, data in H.edges(data=True):
        rel = data.get('relation', 'unknown')
        edge_types[rel] = edge_types.get(rel, 0) + 1
    
    print(f"\nEdge types:")
    for rel, count in edge_types.items():
        print(f"  {rel}: {count}")
    
    # Find parent-child relationships
    print(f"\nSample parent-child relationships:")
    parent_child_count = 0
    for u, v, data in H.edges(data=True):
        if data['relation'] == 'parent_of':
            parent_level = H.nodes[u].get('level', '?')
            child_level = H.nodes[v].get('level', '?')
            print(f"  {u} (L{parent_level}) -> {v} (L{child_level})")
            parent_child_count += 1
            if parent_child_count >= 3:
                break
    print()


def example_cross_scale_queries():
    """Query across different scales to find relationships."""
    print("=" * 60)
    print("Example 4: Cross-Scale Spatial Queries")
    print("=" * 60)
    
    grid = DGGSS2()
    
    # Two locations in Boston
    loc1 = {"lat": 42.3601, "lon": -71.0589, "name": "Boston Common"}
    loc2 = {"lat": 42.3605, "lon": -71.0596, "name": "State House"}
    
    # Find cells at different levels
    for level in [10, 12, 14]:
        token1 = grid.latlon_to_token(loc1["lat"], loc1["lon"], level=level)
        token2 = grid.latlon_to_token(loc2["lat"], loc2["lon"], level=level)
        same_cell = token1 == token2
        
        print(f"Level {level}:")
        print(f"  {loc1['name']}: {token1}")
        print(f"  {loc2['name']}: {token2}")
        print(f"  Same cell? {same_cell}")
        
        if not same_cell:
            # Check if they are neighbors
            neighbors1 = grid.neighbors(token1, ring=1)
            is_neighbor = token2 in neighbors1
            print(f"  Neighbors? {is_neighbor}")
        print()
    
    # Find common ancestor
    token1_fine = grid.latlon_to_token(loc1["lat"], loc1["lon"], level=14)
    token2_fine = grid.latlon_to_token(loc2["lat"], loc2["lon"], level=14)
    
    print("Finding common ancestor (coarsest level where they share the same cell):")
    for level in range(13, -1, -1):
        p1 = grid.parent(token1_fine, parent_level=level)
        p2 = grid.parent(token2_fine, parent_level=level)
        if p1 == p2:
            print(f"  Common ancestor at level {level}: {p1}")
            break
    print()


def example_entity_attachment_multi_scale():
    """Attach entities and analyze at multiple scales."""
    print("=" * 60)
    print("Example 5: Entity Attachment with Multi-Scale Analysis")
    print("=" * 60)
    
    # Entities at different locations
    entities = [
        {"id": "mit", "lat": 42.3601, "lon": -71.0942, "name": "MIT"},
        {"id": "harvard", "lat": 42.3770, "lon": -71.1167, "name": "Harvard"},
        {"id": "bu", "lat": 42.3505, "lon": -71.1054, "name": "Boston University"},
        {"id": "northeastern", "lat": 42.3398, "lon": -71.0892, "name": "Northeastern"},
    ]
    
    grid = DGGSS2(level=13)
    
    # Attach entities at level 13
    G = grid.attach_entities(entities, level=13)
    
    print(f"Entity-cell graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Count node types
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'entity']
    cell_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'cell']
    
    print(f"  Entity nodes: {len(entity_nodes)}")
    print(f"  Cell nodes: {len(cell_nodes)}")
    
    # Now build hierarchical graph to see spatial relationships
    cell_tokens = [n for n in G.nodes() if G.nodes[n].get('type') == 'cell']
    H = grid.build_hierarchical_graph(cell_tokens, include_parents=True)
    
    print(f"\nHierarchical cell graph:")
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")
    
    # Find which entities share parent cells at level 11
    print(f"\nEntities grouped by parent cell (level 11):")
    parent_groups = {}
    for ent in entities:
        token = grid.latlon_to_token(ent["lat"], ent["lon"], level=13)
        parent = grid.parent(token, parent_level=11)
        if parent not in parent_groups:
            parent_groups[parent] = []
        parent_groups[parent].append(ent["name"])
    
    for parent_token, ent_names in parent_groups.items():
        print(f"  {parent_token}: {', '.join(ent_names)}")
    print()


def example_spatial_topology_relations():
    """Demonstrate spatial topology relationships between entities."""
    print("=" * 60)
    print("Example 6: Spatial Topology Relationships")
    print("=" * 60)
    
    grid = DGGSS2(level=12)
    
    # Define some entities
    entities = [
        {"id": "A", "lat": 42.3601, "lon": -71.0589, "name": "Location A"},
        {"id": "B", "lat": 42.3605, "lon": -71.0596, "name": "Location B (nearby)"},
        {"id": "C", "lat": 42.4501, "lon": -71.2589, "name": "Location C (far)"},
    ]
    
    # Check topology between pairs
    print("Cell-based topology relationships (level 12):")
    token_a = grid.latlon_to_token(entities[0]["lat"], entities[0]["lon"], level=12)
    token_b = grid.latlon_to_token(entities[1]["lat"], entities[1]["lon"], level=12)
    token_c = grid.latlon_to_token(entities[2]["lat"], entities[2]["lon"], level=12)
    
    rel_ab = grid.spatial_relation(token_a, token_b)
    rel_ac = grid.spatial_relation(token_a, token_c)
    
    print(f"  A <-> B: {rel_ab}")
    print(f"  A <-> C: {rel_ac}")
    
    # Test at different levels
    print("\nTopology at different scales:")
    for level in [10, 12, 14]:
        token_a_l = grid.latlon_to_token(entities[0]["lat"], entities[0]["lon"], level=level)
        token_b_l = grid.latlon_to_token(entities[1]["lat"], entities[1]["lon"], level=level)
        rel = grid.spatial_relation(token_a_l, token_b_l)
        print(f"  Level {level}: A <-> B = {rel}")
    print()


def example_directional_analysis():
    """Demonstrate directional relationship analysis."""
    print("=" * 60)
    print("Example 7: Directional Relationships")
    print("=" * 60)
    
    grid = DGGSS2()
    
    # Reference point (Boston)
    ref_lat, ref_lon = 42.3601, -71.0589
    
    # Locations in different directions
    locations = [
        {"name": "North", "lat": 42.5, "lon": -71.0589},
        {"name": "South", "lat": 42.2, "lon": -71.0589},
        {"name": "East", "lat": 42.3601, "lon": -70.8},
        {"name": "West", "lat": 42.3601, "lon": -71.3},
        {"name": "Northeast", "lat": 42.5, "lon": -70.8},
        {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    ]
    
    print("Directions from Boston:")
    for loc in locations:
        bearing = grid.bearing(ref_lat, ref_lon, loc["lat"], loc["lon"])
        direction = grid.cardinal_direction(ref_lat, ref_lon, loc["lat"], loc["lon"])
        distance = grid.distance_km(ref_lat, ref_lon, loc["lat"], loc["lon"])
        print(f"  {loc['name']:15s}: {direction:3s} (bearing: {bearing:6.1f}°, distance: {distance:7.1f} km)")
    print()


def example_proximity_queries():
    """Demonstrate proximity-based spatial queries."""
    print("=" * 60)
    print("Example 8: Proximity Queries and Relationship Analysis")
    print("=" * 60)
    
    # Universities in Boston area
    entities = [
        {"id": "mit", "lat": 42.3601, "lon": -71.0942, "name": "MIT"},
        {"id": "harvard", "lat": 42.3770, "lon": -71.1167, "name": "Harvard"},
        {"id": "bu", "lat": 42.3505, "lon": -71.1054, "name": "Boston University"},
        {"id": "northeastern", "lat": 42.3398, "lon": -71.0892, "name": "Northeastern"},
        {"id": "tufts", "lat": 42.4075, "lon": -71.1190, "name": "Tufts"},
    ]
    
    grid = DGGSS2(level=13)
    
    # Find entities near MIT
    print("Entities within 5 km of MIT:")
    nearby = grid.find_entities_in_range(
        entities, 
        center_lat=42.3601, 
        center_lon=-71.0942,
        radius_km=5,
        level=13
    )
    for ent in nearby:
        print(f"  {ent['name']:20s}: {ent['distance_km']:5.2f} km to the {ent['direction']}")
    
    # Analyze all pairwise relationships
    print("\nPairwise relationship analysis:")
    relationships = analyze_entity_relationships(
        entities,
        level=13,
        distance_threshold_km=3.0
    )
    
    for rel in relationships[:5]:  # Show first 5
        print(f"  {rel['entity1_id']:12s} <-> {rel['entity2_id']:12s}: "
              f"{rel['topology']:8s}, {rel['direction']:3s}, "
              f"{rel['distance_km']:5.2f} km, {rel['proximity']:8s}")
    
    print(f"\n  Total relationships analyzed: {len(relationships)}")
    
    # Group by topology type
    topology_counts = {}
    for rel in relationships:
        topo = rel['topology']
        topology_counts[topo] = topology_counts.get(topo, 0) + 1
    
    print("\nTopology distribution:")
    for topo, count in topology_counts.items():
        print(f"  {topo}: {count}")
    print()


if __name__ == "__main__":
    example_basic_cell_operations()
    example_hierarchical_relationships()
    example_multi_scale_graph()
    example_cross_scale_queries()
    example_entity_attachment_multi_scale()
    example_spatial_topology_relations()
    example_directional_analysis()
    example_proximity_queries()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
