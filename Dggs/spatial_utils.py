"""
DGGS Spatial Utilities Module

Provides core utility functions for spatial discretization:
- Statistical aggregation functions (shared across point, polyline, polygon, raster modules)
- Geometric algorithms (point-in-polygon, segment intersection, etc.)
- Helper functions for spatial operations

This module consolidates common functionality previously split between
geometry.py and discretize.py, reducing code duplication.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, Union
from s2sphere import CellId, LatLngRect, Cell, LatLng
from .dggs import DggsS2
import statistics
import math


####################################################################
# Statistical Aggregation Functions
# Used by point.py, polyline.py, polygon.py, raster.py modules
####################################################################

def aggregate_values(
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    method: str = 'mean'
) -> Optional[float]:
    """Aggregate a list of values using specified method.
    
    Core aggregation function used across all spatial data types.
    
    Args:
        values: List of numeric values to aggregate
        weights: Optional weights for weighted aggregations
        method: Aggregation method
            - 'mean': Simple average
            - 'weighted_mean': Weight-weighted average
            - 'sum': Sum of values
            - 'max': Maximum value
            - 'min': Minimum value
            - 'median': Median value
            - 'std': Standard deviation
            - 'count': Count of values
            - 'mode': Most frequent value
    
    Returns:
        Aggregated value or None if values is empty
    
    Example:
        >>> aggregate_values([1, 2, 3, 4, 5], method='mean')
        3.0
        >>> aggregate_values([1, 2, 3], weights=[1, 2, 1], method='weighted_mean')
        2.0
    """
    if not values:
        return None
    
    if method == 'count':
        return float(len(values))
    
    if method == 'weighted_mean':
        if weights is None or len(weights) != len(values):
            return statistics.mean(values)
        total_weight = sum(weights)
        if total_weight == 0:
            return statistics.mean(values)
        return sum(v * w for v, w in zip(values, weights)) / total_weight
    
    elif method == 'mean':
        return statistics.mean(values)
    
    elif method == 'sum':
        return sum(values)
    
    elif method == 'max':
        return max(values)
    
    elif method == 'min':
        return min(values)
    
    elif method == 'median':
        return statistics.median(values)
    
    elif method == 'std':
        return statistics.stdev(values) if len(values) > 1 else 0.0
    
    elif method == 'mode':
        return statistics.mode(values) if values else None
    
    else:
        # Default to mean
        return statistics.mean(values)


def discretize_aggregate(
    data: Sequence[Dict[str, Any]],
    value_fields: List[str],
    level: int = 12,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
    entity_id_key: str = "id"
) -> Dict[str, Dict[str, Any]]:
    """Aggregate data values within each DGGS cell using statistical functions.
    
    Generic aggregation function used by domain-specific modules.
    When multiple observations fall into the same cell, aggregate their values
    using specified functions.
    
    Args:
        data: Sequence of observations with 'lat', 'lon' and value fields
        value_fields: List of field names containing values to aggregate
        level: Target S2 cell level
        agg_funcs: Dict mapping field -> aggregation function
            String options: 'mean', 'sum', 'min', 'max', 'median', 'std', 'count'
            Or callable: custom aggregation function(values: List[float]) -> float
        entity_id_key: Key for observation identifier
    
    Returns:
        Dictionary mapping cell_token -> {field_agg: value, count: n, entity_ids: [...]}
    
    Example:
        data = [
            {"id": "obs1", "lat": 40.7, "lon": -74.0, "temp": 25.5, "humidity": 60},
            {"id": "obs2", "lat": 40.71, "lon": -74.01, "temp": 24.8, "humidity": 65}
        ]
        result = discretize_aggregate(
            data, 
            ["temp", "humidity"], 
            agg_funcs={"temp": "mean", "humidity": "max"}
        )
    """
    if agg_funcs is None:
        agg_funcs = {field: 'mean' for field in value_fields}
    
    grid = DGGSS2(level=level)
    cell_data: Dict[str, Dict[str, List[Any]]] = {}
    
    # Group data by cell
    for obs in data:
        lat, lon = obs.get("lat"), obs.get("lon")
        if lat is None or lon is None:
            continue
        
        token = grid.latlon_to_token(lat, lon, level)
        
        if token not in cell_data:
            cell_data[token] = {field: [] for field in value_fields}
            cell_data[token]["_entity_ids"] = []
        
        for field in value_fields:
            if field in obs and obs[field] is not None:
                try:
                    cell_data[token][field].append(float(obs[field]))
                except (ValueError, TypeError):
                    continue
        
        entity_id = obs.get(entity_id_key, obs.get("name"))
        if entity_id:
            cell_data[token]["_entity_ids"].append(entity_id)
    
    # Aggregate values in each cell
    result = {}
    for token, fields in cell_data.items():
        result[token] = {
            "count": len(fields["_entity_ids"]),
            "entity_ids": fields["_entity_ids"]
        }
        
        for field in value_fields:
            values = fields[field]
            if not values:
                continue
            
            agg_func = agg_funcs.get(field, 'mean')
            
            if callable(agg_func):
                result[token][f"{field}"] = agg_func(values)
            else:
                result[token][f"{field}_{agg_func}"] = aggregate_values(values, method=agg_func)
    
    return result


def discretize_weighted_aggregate(
    data: Sequence[Dict[str, Any]],
    value_fields: List[str],
    weight_field: str,
    level: int = 12,
    agg_method: str = 'weighted_mean',
    entity_id_key: str = "id"
) -> Dict[str, Dict[str, Any]]:
    """Aggregate data with weighted values.
    
    Similar to discretize_aggregate but uses a weight field for weighted aggregation.
    Used when observations have different importance/reliability.
    
    Args:
        data: Sequence of observations with 'lat', 'lon', value fields, and weight field
        value_fields: List of field names containing values to aggregate
        weight_field: Field name containing weights
        level: Target S2 cell level
        agg_method: Aggregation method ('weighted_mean', 'mean', 'sum', etc.)
        entity_id_key: Key for observation identifier
    
    Returns:
        Dictionary mapping cell_token -> {field: weighted_value, count: n}
    
    Example:
        data = [
            {"id": "obs1", "lat": 40.7, "lon": -74.0, "temp": 25.5, "quality": 0.9},
            {"id": "obs2", "lat": 40.7, "lon": -74.0, "temp": 24.8, "quality": 0.7}
        ]
        result = discretize_weighted_aggregate(
            data,
            ["temp"],
            weight_field="quality"
        )
    """
    grid = DGGSS2(level=level)
    cell_data: Dict[str, Dict[str, List[float]]] = {}
    
    # Group data by cell
    for obs in data:
        lat, lon = obs.get("lat"), obs.get("lon")
        if lat is None or lon is None:
            continue
        
        weight = obs.get(weight_field)
        if weight is None:
            weight = 1.0
        
        token = grid.latlon_to_token(lat, lon, level)
        
        if token not in cell_data:
            cell_data[token] = {field: [] for field in value_fields}
            cell_data[token]["_weights"] = []
            cell_data[token]["_entity_ids"] = []
        
        for field in value_fields:
            if field in obs and obs[field] is not None:
                try:
                    cell_data[token][field].append(float(obs[field]))
                except (ValueError, TypeError):
                    continue
        
        cell_data[token]["_weights"].append(float(weight))
        
        entity_id = obs.get(entity_id_key, obs.get("name"))
        if entity_id:
            cell_data[token]["_entity_ids"].append(entity_id)
    
    # Aggregate with weights
    result = {}
    for token, fields in cell_data.items():
        result[token] = {
            "count": len(fields["_entity_ids"]),
            "entity_ids": fields["_entity_ids"]
        }
        
        weights = fields["_weights"]
        
        for field in value_fields:
            values = fields[field]
            if not values:
                continue
            
            if agg_method == 'weighted_mean':
                result[token][field] = aggregate_values(values, weights, 'weighted_mean')
            else:
                result[token][field] = aggregate_values(values, method=agg_method)
    
    return result


####################################################################
# Geometric Algorithms
# Point-in-polygon, segment intersection, etc.
####################################################################

def point_in_ring(point: Tuple[float, float], ring: List[Tuple[float, float]]) -> bool:
    """Test if point is inside a polygon ring using ray casting algorithm.
    
    Args:
        point: (lat, lon) tuple
        ring: List of (lat, lon) tuples forming closed polygon ring
    
    Returns:
        True if point is inside ring, False otherwise
    
    Note:
        Uses ray casting algorithm - counts intersections with edges
    """
    x, y = point[1], point[0]  # lon, lat
    inside = False
    n = len(ring)
    
    for i in range(n):
        lat_i, lon_i = ring[i]
        lat_j, lon_j = ring[(i + 1) % n]
        xi, yi = lon_i, lat_i
        xj, yj = lon_j, lat_j
        
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
    
    return inside


def point_in_polygon(
    point: Tuple[float, float],
    outer_ring: List[Tuple[float, float]],
    holes: Optional[List[List[Tuple[float, float]]]] = None
) -> bool:
    """Test if point is inside polygon (considering holes).
    
    Args:
        point: (lat, lon) tuple
        outer_ring: Outer boundary of polygon as list of (lat, lon) tuples
        holes: Optional list of hole rings (inner boundaries)
    
    Returns:
        True if point is inside polygon and not in any hole
    """
    if not point_in_ring(point, outer_ring):
        return False
    
    if holes:
        for hole in holes:
            if point_in_ring(point, hole):
                return False
    
    return True


def segments_intersect(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float]
) -> bool:
    """Check if two line segments intersect.
    
    Args:
        a1, a2: Endpoints of first segment as (lat, lon) tuples
        b1, b2: Endpoints of second segment as (lat, lon) tuples
    
    Returns:
        True if segments intersect or touch
    """
    def orient(p, q, r):
        """Calculate orientation of ordered triplet (p, q, r)."""
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    
    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        return (min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and
                min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12)
    
    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    
    # General case
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    
    # Collinear cases
    if abs(o1) < 1e-12 and on_segment(a1, b1, a2):
        return True
    if abs(o2) < 1e-12 and on_segment(a1, b2, a2):
        return True
    if abs(o3) < 1e-12 and on_segment(b1, a1, b2):
        return True
    if abs(o4) < 1e-12 and on_segment(b1, a2, b2):
        return True
    
    return False


def interpolate_line_segment(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    step_km: float
) -> List[Tuple[float, float]]:
    """Interpolate points along a line segment.
    
    Used by polyline discretization to densify paths for better cell coverage.
    
    Args:
        lat1, lon1: Start point coordinates
        lat2, lon2: End point coordinates
        step_km: Distance between interpolated points in kilometers
    
    Returns:
        List of (lat, lon) tuples along the segment
    """
    grid = DGGSS2()
    dist = grid.distance_km(lat1, lon1, lat2, lon2)
    n = max(1, int(dist / max(step_km, 0.001)))
    
    points = []
    for i in range(n + 1):
        t = i / n
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        points.append((lat, lon))
    
    return points


def cell_vertices_latlon(cell: Cell) -> List[Tuple[float, float]]:
    """Get S2 cell vertices as (lat, lon) tuples.
    
    Args:
        cell: S2 Cell object
    
    Returns:
        List of 4 vertices as (lat, lon) tuples
    """
    vertices = []
    for i in range(4):
        point = cell.get_vertex(i)
        ll = LatLng.from_point(point)
        vertices.append((ll.lat().degrees, ll.lng().degrees))
    return vertices


def ring_edges(ring: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Extract edges from a polygon ring.
    
    Args:
        ring: List of (lat, lon) tuples forming a closed ring
    
    Returns:
        List of edges as ((lat1, lon1), (lat2, lon2)) tuples
    """
    return list(zip(ring, ring[1:] + [ring[0]]))


####################################################################
# Strict Polygon Discretization
# Accurate polygon-to-cell mapping with hole support
####################################################################

def discretize_polygon_strict(
    polygon_coords: Sequence[Sequence[Sequence[float]]],
    level: int = 12,
    max_cells: int = 1000
) -> List[str]:
    """Strict polygon discretization with hole support.
    
    Accurately determines which cells intersect or are contained by the polygon,
    including support for holes (inner rings).
    
    This function is computationally expensive but provides accurate results.
    For large polygons, consider using bbox approximation instead.
    
    Args:
        polygon_coords: GeoJSON polygon coordinates [[outer_ring], [hole1], [hole2], ...]
            Each ring is [[lon1, lat1], [lon2, lat2], ...]
        level: Target S2 cell level
        max_cells: Maximum number of candidate cells to test
    
    Returns:
        List of cell tokens that intersect or are contained by the polygon
    
    Algorithm:
        1. Compute bounding box and get candidate cells
        2. For each candidate cell, test:
           - Any cell vertex inside polygon → include
           - Any polygon vertex inside cell → include
           - Cell edge intersects polygon edge → include
    
    Example:
        polygon = [
            [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon1, lat1]],  # outer ring
            [[lon4, lat4], [lon5, lat5], [lon6, lat6], [lon4, lat4]]   # hole
        ]
        tokens = discretize_polygon_strict(polygon, level=13)
    """
    # Convert to (lat, lon) format
    rings_latlon: List[List[Tuple[float, float]]] = [
        [(p[1], p[0]) for p in ring] for ring in polygon_coords
    ]
    outer = rings_latlon[0]
    holes = rings_latlon[1:] if len(rings_latlon) > 1 else []
    
    # Compute bounding box
    lats = [lat for lat, _ in outer]
    lons = [lon for _, lon in outer]
    sw = (min(lats), min(lons))
    ne = (max(lats), max(lons))
    
    # Get candidate cells
    grid = DGGSS2(level=level, max_cells=max_cells)
    candidates = grid.cover_rectangle(sw, ne, level=level, max_cells=max_cells)
    
    # Collect all polygon edges (outer + holes)
    polygon_edges = ring_edges(outer)
    for hole in holes:
        polygon_edges += ring_edges(hole)
    
    # Test each candidate cell
    tokens: List[str] = []
    for tok in candidates:
        c = Cell(CellId.from_token(tok))
        rect = c.get_rect_bound()
        
        # Quick reject if outside bbox
        if (rect.lat_lo().degrees > ne[0] or rect.lat_hi().degrees < sw[0] or
            rect.lng_lo().degrees > ne[1] or rect.lng_hi().degrees < sw[1]):
            continue
        
        verts = cell_vertices_latlon(c)
        
        # Test 1: Any cell vertex inside polygon → include cell
        any_vert_inside = any(point_in_polygon(v, outer, holes) for v in verts)
        if any_vert_inside:
            tokens.append(tok)
            continue
        
        # Test 2: Any polygon vertex inside cell → include cell
        def point_in_rect(latlon: Tuple[float, float], rect: LatLngRect) -> bool:
            lat, lon = latlon
            return (rect.lat_lo().degrees - 1e-12 <= lat <= rect.lat_hi().degrees + 1e-12 and
                    rect.lng_lo().degrees - 1e-12 <= lon <= rect.lng_hi().degrees + 1e-12)
        
        if any(point_in_rect(v, rect) for v in outer):
            tokens.append(tok)
            continue
        
        # Test 3: Cell edge intersects polygon edge → include cell
        cell_edges = ring_edges(verts)
        intersect = False
        for e1 in cell_edges:
            for e2 in polygon_edges:
                if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                    intersect = True
                    break
            if intersect:
                break
        
        if intersect:
            tokens.append(tok)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    
    return unique


####################################################################
# Utility Functions
####################################################################

def calculate_distance_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """Calculate distance between two points in kilometers.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    grid = DGGSS2()
    return grid.distance_km(lat1, lon1, lat2, lon2)


def calculate_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """Calculate bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Start point coordinates
        lat2, lon2: End point coordinates
    
    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


def calculate_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate centroid of a set of points.
    
    Args:
        points: List of (lat, lon) tuples
    
    Returns:
        Centroid as (lat, lon) tuple
    """
    if not points:
        return (0.0, 0.0)
    
    avg_lat = statistics.mean(p[0] for p in points)
    avg_lon = statistics.mean(p[1] for p in points)
    
    return (avg_lat, avg_lon)


def calculate_bbox(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calculate bounding box of a set of points.
    
    Args:
        points: List of (lat, lon) tuples
    
    Returns:
        ((min_lat, min_lon), (max_lat, max_lon))
    """
    if not points:
        return ((0.0, 0.0), (0.0, 0.0))
    
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    
    return ((min(lats), min(lons)), (max(lats), max(lons)))
