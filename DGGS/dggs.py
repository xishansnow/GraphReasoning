"""S2-based Discrete Global Grid System utilities (hard-moved into DGGS package)."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import networkx as nx
from s2sphere import Angle, Cap, CellId, LatLng, LatLngRect, RegionCoverer, Cell


EARTH_RADIUS_KM = 6371.0088


@dataclass
class S2GridConfig:
    level: int = 12
    max_cells: int = 200


class DGGSS2:
    def __init__(self, level: int = 12, max_cells: int = 200):            
        self.config = S2GridConfig(level=level, max_cells=max_cells)

    def latlon_to_cell(self, lat: float, lon: float, level: Optional[int] = None) -> CellId:
        lvl = self._level(level)
        ll = LatLng.from_degrees(lat, lon)
        return CellId.from_lat_lng(ll).parent(lvl)

    def latlon_to_token(self, lat: float, lon: float, level: Optional[int] = None) -> str:
        return self.latlon_to_cell(lat, lon, level).to_token()

    def token_to_cell(self, token: str) -> CellId:
        return CellId.from_token(token)

    def cell_center(self, token: str) -> Tuple[float, float]:
        ll = self.token_to_cell(token).to_lat_lng()
        return ll.lat().degrees, ll.lng().degrees

    def parent(self, token: str, parent_level: Optional[int] = None) -> str:
        cell = self.token_to_cell(token)
        if parent_level is None:
            parent_level = max(0, cell.level() - 1)
        if parent_level >= cell.level():
            raise ValueError(f"Parent level {parent_level} must be < current level {cell.level()}")
        return cell.parent(parent_level).to_token()

    def children(self, token: str, child_level: Optional[int] = None) -> List[str]:
        cell = self.token_to_cell(token)
        if child_level is None:
            child_level = min(30, cell.level() + 1)
        if child_level <= cell.level():
            raise ValueError(f"Child level {child_level} must be > current level {cell.level()}")
        children = []
        for child_id in cell.children():
            if child_id.level() < child_level:
                for descendant_id in child_id.children():
                    if descendant_id.level() == child_level:
                        children.append(descendant_id.to_token())
                    elif descendant_id.level() < child_level:
                        children.extend(self.children(descendant_id.to_token(), child_level))
            elif child_id.level() == child_level:
                children.append(child_id.to_token())
        return children

    def neighbors(self, token: str, ring: int = 1) -> List[str]:
        origin = self.token_to_cell(token)
        seen = {origin.id()}
        frontier = [origin]
        for _ in range(ring):
            next_frontier: List[CellId] = []
            for cell in frontier:
                adj = cell.get_edge_neighbors()
                for n in adj:
                    if n.id() not in seen and n.level() == origin.level():
                        seen.add(n.id())
                        next_frontier.append(n)
            frontier = next_frontier
        seen.remove(origin.id())
        return [CellId(x).to_token() for x in seen]

    def cover_rectangle(self, sw: Tuple[float, float], ne: Tuple[float, float], level: Optional[int] = None, max_cells: Optional[int] = None) -> List[str]:
        lvl = self._level(level)
        coverer = self._coverer(max_cells)
        rect = LatLngRect.from_point_pair(LatLng.from_degrees(sw[0], sw[1]), LatLng.from_degrees(ne[0], ne[1]))
        coverer.min_level = coverer.max_level = lvl
        cells = coverer.get_covering(rect)
        return [c.to_token() for c in cells]

    def cover_cap(self, center_lat: float, center_lon: float, radius_km: float, level: Optional[int] = None, max_cells: Optional[int] = None) -> List[str]:
        lvl = self._level(level)
        coverer = self._coverer(max_cells)
        coverer.min_level = coverer.max_level = lvl
        ll = LatLng.from_degrees(center_lat, center_lon)
        cap = Cap.from_axis_angle(ll.to_point(), Angle.from_radians(radius_km / EARTH_RADIUS_KM))
        cells = coverer.get_covering(cap)
        return [c.to_token() for c in cells]

    def build_cell_graph(self, cell_tokens: Iterable[str]) -> nx.Graph:
        tokens = set(cell_tokens)
        G = nx.Graph()
        for token in tokens:
            cell = self.token_to_cell(token)
            lat, lon = self.cell_center(token)
            G.add_node(token, type="cell", level=cell.level(), lat=lat, lon=lon)
        for token in tokens:
            for n in self.neighbors(token, ring=1):
                if n in tokens:
                    G.add_edge(token, n, relation="adjacent")
        return G

    def build_hierarchical_graph(self, cell_tokens: Iterable[str], include_parents: bool = True, include_children: bool = False) -> nx.DiGraph:
        tokens = set(cell_tokens)
        G = nx.DiGraph()
        for token in tokens:
            cell = self.token_to_cell(token)
            lat, lon = self.cell_center(token)
            G.add_node(token, type="cell", level=cell.level(), lat=lat, lon=lon)
        if include_parents:
            for token in list(tokens):
                cell = self.token_to_cell(token)
                if cell.level() > 0:
                    parent_token = self.parent(token)
                    if parent_token not in G:
                        p_cell = self.token_to_cell(parent_token)
                        p_lat, p_lon = self.cell_center(parent_token)
                        G.add_node(parent_token, type="cell", level=p_cell.level(), lat=p_lat, lon=p_lon)
                    G.add_edge(parent_token, token, relation="parent_of")
                    G.add_edge(token, parent_token, relation="child_of")
        if include_children:
            for token in list(tokens):
                cell = self.token_to_cell(token)
                if cell.level() < 30:
                    for child_token in self.children(token, cell.level() + 1):
                        if child_token not in G:
                            c_cell = self.token_to_cell(child_token)
                            c_lat, c_lon = self.cell_center(child_token)
                            G.add_node(child_token, type="cell", level=c_cell.level(), lat=c_lat, lon=c_lon)
                        G.add_edge(token, child_token, relation="parent_of")
                        G.add_edge(child_token, token, relation="child_of")
        for token in G.nodes():
            if G.nodes[token].get("type") == "cell":
                for n in self.neighbors(token, ring=1):
                    if n in G:
                        G.add_edge(token, n, relation="adjacent")
                        G.add_edge(n, token, relation="adjacent")
        return G

    def attach_entities(self, entities: Sequence[Dict[str, Any]], level: Optional[int] = None, entity_id_key: str = "id") -> nx.Graph:
        tokens: List[str] = []
        G = nx.Graph()
        for idx, ent in enumerate(entities):
            lat = ent.get("lat")
            lon = ent.get("lon")
            if lat is None or lon is None:
                continue
            token = self.latlon_to_token(lat, lon, level)
            tokens.append(token)
            ent_id = ent.get(entity_id_key) or ent.get("name") or f"entity_{idx}"
            G.add_node(ent_id, type="entity", **{k: v for k, v in ent.items() if k not in {"lat", "lon"}})
            G.add_node(token, type="cell")
            G.add_edge(ent_id, token, relation="belongs_to")
        cell_graph = self.build_cell_graph(tokens)
        return nx.compose(G, cell_graph)

    def _coverer(self, max_cells: Optional[int]) -> RegionCoverer:
        coverer = RegionCoverer()
        coverer.max_cells = max_cells or self.config.max_cells
        return coverer

    def _level(self, level: Optional[int]) -> int:
        return self.config.level if level is None else level

    def distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        ll1 = LatLng.from_degrees(lat1, lon1)
        ll2 = LatLng.from_degrees(lat2, lon2)
        angle = ll1.get_distance(ll2)
        return angle.radians * EARTH_RADIUS_KM

    def bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        return bearing_deg

    def cardinal_direction(self, lat1: float, lon1: float, lat2: float, lon2: float) -> str:
        bearing = self.bearing(lat1, lon1, lat2, lon2)
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = round(bearing / 45) % 8
        return directions[index]

    def spatial_relation(self, token1: str, token2: str) -> str:
        cell1 = self.token_to_cell(token1)
        cell2 = self.token_to_cell(token2)
        if cell1.id() == cell2.id():
            return 'equal'
        if cell1.contains(cell2):
            return 'contains'
        if cell2.contains(cell1):
            return 'within'
        if cell1.level() == cell2.level():
            neighbors1 = self.neighbors(token1, ring=1)
            if token2 in neighbors1:
                return 'adjacent'
        return 'disjoint'

    def entity_relation(self, entity1: Dict[str, Any], entity2: Dict[str, Any], level: Optional[int] = None, distance_threshold_km: Optional[float] = None) -> Dict[str, Any]:
        lat1, lon1 = entity1.get('lat'), entity1.get('lon')
        lat2, lon2 = entity2.get('lat'), entity2.get('lon')
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            raise ValueError("Both entities must have 'lat' and 'lon' attributes")
        distance = self.distance_km(lat1, lon1, lat2, lon2)
        direction = self.cardinal_direction(lat1, lon1, lat2, lon2)
        token1 = self.latlon_to_token(lat1, lon1, level)
        token2 = self.latlon_to_token(lat2, lon2, level)
        topology = self.spatial_relation(token1, token2)
        proximity = 'distant'
        if distance_threshold_km:
            if distance < distance_threshold_km:
                proximity = 'near'
            elif distance < distance_threshold_km * 3:
                proximity = 'moderate'
        return {
            'topology': topology,
            'direction': direction,
            'distance_km': round(distance, 3),
            'proximity': proximity,
            'cell1': token1,
            'cell2': token2,
        }

    def find_entities_in_range(self, entities: Sequence[Dict[str, Any]], center_lat: float, center_lon: float, radius_km: float, level: Optional[int] = None, entity_id_key: str = "id") -> List[Dict[str, Any]]:
        cells_in_range = set(self.cover_cap(center_lat, center_lon, radius_km, level))
        results = []
        for idx, ent in enumerate(entities):
            lat = ent.get('lat')
            lon = ent.get('lon')
            if lat is None or lon is None:
                continue
            ent_token = self.latlon_to_token(lat, lon, level)
            if ent_token in cells_in_range:
                distance = self.distance_km(center_lat, center_lon, lat, lon)
                if distance <= radius_km:
                    direction = self.cardinal_direction(center_lat, center_lon, lat, lon)
                    ent_copy = dict(ent)
                    ent_copy['distance_km'] = round(distance, 3)
                    ent_copy['direction'] = direction
                    results.append(ent_copy)
        results.sort(key=lambda x: x['distance_km'])
        return results


def entity_to_cell_tokens(entities: Sequence[Dict[str, Any]], level: int = 12, entity_id_key: str = "id") -> Dict[str, str]:
    grid = DGGSS2(level=level)
    out: Dict[str, str] = {}
    for idx, ent in enumerate(entities):
        lat = ent.get("lat")
        lon = ent.get("lon")
        if lat is None or lon is None:
            continue
        ent_id = ent.get(entity_id_key) or ent.get("name") or f"entity_{idx}"
        out[ent_id] = grid.latlon_to_token(lat, lon)
    return out


def analyze_entity_relationships(entities: Sequence[Dict[str, Any]], level: int = 12, distance_threshold_km: float = 10.0, entity_id_key: str = "id") -> List[Dict[str, Any]]:
    grid = DGGSS2(level=level)
    relationships = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            ent1 = entities[i]
            ent2 = entities[j]
            lat1, lon1 = ent1.get('lat'), ent1.get('lon')
            lat2, lon2 = ent2.get('lat'), ent2.get('lon')
            if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                continue
            relation = grid.entity_relation(ent1, ent2, level, distance_threshold_km)
            ent1_id = ent1.get(entity_id_key) or ent1.get("name") or f"entity_{i}"
            ent2_id = ent2.get(entity_id_key) or ent2.get("name") or f"entity_{j}"
            relationships.append({'entity1_id': ent1_id,'entity2_id': ent2_id,**relation,})
    return relationships


def build_topology_enriched_graph(entities: Sequence[Dict[str, Any]], level: int = 12, include_cells: bool = True, include_hierarchy: bool = True, distance_threshold_km: float = 5.0, entity_id_key: str = "id") -> nx.MultiDiGraph:
    grid = DGGSS2(level=level)
    G = nx.MultiDiGraph()
    for idx, ent in enumerate(entities):
        ent_id = ent.get(entity_id_key) or ent.get("name") or f"entity_{idx}"
        G.add_node(ent_id, type="entity", **ent)
    cell_tokens = []
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
    rels = analyze_entity_relationships(entities, level=level, distance_threshold_km=distance_threshold_km, entity_id_key=entity_id_key)
    for rel in rels:
        u = rel["entity1_id"]
        v = rel["entity2_id"]
        G.add_edge(u, v, type="topology", relation=rel["topology"])
        G.add_edge(u, v, type="direction", direction=rel["direction"], distance_km=rel["distance_km"], proximity=rel["proximity"]) 
    return G
