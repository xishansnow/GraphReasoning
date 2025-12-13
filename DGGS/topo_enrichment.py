from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import networkx as nx
from .dggs import DGGSS2, analyze_entity_relationships


def build_topology_enriched_graph(
    entities: Sequence[Dict[str, Any]],
    level: int = 12,
    include_cells: bool = True,
    include_hierarchy: bool = True,
    distance_threshold_km: float = 5.0,
    entity_id_key: str = "id",
) -> nx.MultiDiGraph:
    grid = DGGSS2(level=level)
    G = nx.MultiDiGraph()

    # Add entity nodes
    for idx, ent in enumerate(entities):
        ent_id = ent.get(entity_id_key) or ent.get("name") or f"entity_{idx}"
        G.add_node(ent_id, type="entity", **ent)

    # Map entities to cells and optionally add cell graph
    cell_tokens: List[str] = []
    for idx, ent in enumerate(entities):
        lat = ent.get("lat")
        lon = ent.get("lon")
        if lat is None or lon is None:
            continue
        ent_id = ent.get(entity_id_key) or ent.get("name") or f"entity_{idx}"
        tok = grid.latlon_to_token(lat, lon, level)
        cell_tokens.append(tok)
        if include_cells:
            cell = grid.token_to_cell(tok)
            c_lat, c_lon = grid.cell_center(tok)
            G.add_node(tok, type="cell", level=cell.level(), lat=c_lat, lon=c_lon)
            G.add_edge(ent_id, tok, relation="belongs_to")

    # Cell adjacency and hierarchy
    if include_cells and cell_tokens:
        base_cell_graph = grid.build_cell_graph(cell_tokens)
        for n, data in base_cell_graph.nodes(data=True):
            if n not in G:
                G.add_node(n, **data)
        for u, v, data in base_cell_graph.edges(data=True):
            G.add_edge(u, v, relation=data.get("relation", "adjacent"))
            G.add_edge(v, u, relation=data.get("relation", "adjacent"))

        if include_hierarchy:
            H = grid.build_hierarchical_graph(cell_tokens, include_parents=True, include_children=False)
            for n, data in H.nodes(data=True):
                if n not in G:
                    G.add_node(n, **data)
            for u, v, data in H.edges(data=True):
                G.add_edge(u, v, relation=data.get("relation", "adjacent"))

    # Entity-to-entity relations
    rels = analyze_entity_relationships(entities, level=level, distance_threshold_km=distance_threshold_km, entity_id_key=entity_id_key)
    for rel in rels:
        u = rel["entity1_id"]
        v = rel["entity2_id"]
        G.add_edge(u, v, type="topology", relation=rel["topology"])
        G.add_edge(u, v, type="direction", direction=rel["direction"], distance_km=rel["distance_km"], proximity=rel["proximity"]) 

    return G
