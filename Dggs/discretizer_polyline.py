"""
DGGS Polyline Discretization Module

Generic discretization framework for polyline/line vector data with attributes.
Supports any line-based spatial data (roads, rivers, pipelines, trails, etc.)
with nested segments/attributes and hierarchical structure.

Key concepts:
- Polyline: Geographic line defined by coordinate sequences
- Segments: Subsections of polylines (with attributes/properties)
- Attributes: Properties associated with segments or entire lines
- Topology: Connectivity, directionality, network relationships

Supported data types:
- Transportation networks (roads, railways, bike paths)
- Hydrological networks (rivers, streams, drainage)
- Utility networks (pipelines, power lines, cables)
- Administrative boundaries (borders, fence lines)
- Ecological corridors (migration routes, habitat connections)
- Trails and paths (hiking, ski, off-road)

Architecture:
    PolylineFeature (base class)
        ├── LineSegment (nested segments with attributes)
        └── Discretization methods (interpolate, sample, buffer)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, Set
from s2sphere import CellId, LatLng, Cell
from .dggs import DggsS2
from .spatial_utils import discretize_aggregate
import statistics
import math


####################################################################
# Generic Polyline Data Models
####################################################################

class LineSegment:
    """Represents a segment/section within a polyline.
    
    Segments are subsections of a polyline, each with:
    - Start and end points
    - Length and direction
    - Attributes (properties/measurements)
    - Optional sub-segments (hierarchical)
    
    Examples:
    - Road segments between intersections
    - River reaches between confluences
    - Pipeline sections between valves
    - Trail segments between waypoints
    """
    
    def __init__(self,
                 segment_id: str,
                 start_point: Tuple[float, float],
                 end_point: Tuple[float, float],
                 attributes: Optional[Dict[str, Any]] = None,
                 weight: float = 1.0,
                 sub_segments: Optional[List['LineSegment']] = None):
        """
        Args:
            segment_id: Unique segment identifier
            start_point: (lat, lon) tuple for segment start
            end_point: (lat, lon) tuple for segment end
            attributes: Dictionary of segment properties
            weight: Relative importance/weight (e.g., traffic volume, flow rate)
            sub_segments: Nested segments (for hierarchical data)
        """
        self.segment_id = segment_id
        self.start_point = start_point
        self.end_point = end_point
        self.attributes = attributes or {}
        self.weight = weight
        self.sub_segments = sub_segments or []
        self.length_km = self._calculate_length()
    
    def _calculate_length(self) -> float:
        """Calculate segment length in kilometers."""
        grid = DGGSS2()
        return grid.distance_km(
            self.start_point[0], self.start_point[1],
            self.end_point[0], self.end_point[1]
        )
    
    def get_attribute(self, attr_name: str, default: Any = None) -> Any:
        """Get attribute value with fallback."""
        return self.attributes.get(attr_name, default)
    
    def set_attribute(self, attr_name: str, value: Any) -> None:
        """Set attribute value."""
        self.attributes[attr_name] = value
    
    def get_midpoint(self) -> Tuple[float, float]:
        """Get segment midpoint coordinates."""
        lat = (self.start_point[0] + self.end_point[0]) / 2
        lon = (self.start_point[1] + self.end_point[1]) / 2
        return (lat, lon)


class PolylineFeature:
    """Generic polyline feature with segments and attributes.
    
    Represents any line-based spatial feature that can contain:
    - Coordinate sequence defining the line
    - Multiple segments (with attributes)
    - Topological properties (direction, connectivity)
    - Hierarchical sub-features
    
    This is the base class for domain-specific implementations like
    roads, rivers, pipelines, trails, etc.
    """
    
    def __init__(self,
                 feature_id: str,
                 coordinates: Sequence[Tuple[float, float]],
                 segments: Optional[List[LineSegment]] = None,
                 attributes: Optional[Dict[str, Any]] = None,
                 feature_type: Optional[str] = None,
                 is_directed: bool = False):
        """
        Args:
            feature_id: Unique identifier for this polyline
            coordinates: List of (lat, lon) tuples defining the line
            segments: List of LineSegment objects
            attributes: Polyline-level attributes (not segment-specific)
            feature_type: Type descriptor (e.g., 'road', 'river', 'pipeline')
            is_directed: Whether line has direction (e.g., one-way road, river flow)
        """
        self.feature_id = feature_id
        self.coordinates = list(coordinates)
        self.segments = segments or []
        self.attributes = attributes or {}
        self.feature_type = feature_type or 'generic_polyline'
        self.is_directed = is_directed
        self.total_length_km = self._calculate_total_length()
        self.start_point = coordinates[0] if coordinates else (0.0, 0.0)
        self.end_point = coordinates[-1] if coordinates else (0.0, 0.0)
    
    def _calculate_total_length(self) -> float:
        """Calculate total line length in kilometers."""
        if len(self.coordinates) < 2:
            return 0.0
        
        grid = DGGSS2()
        total = 0.0
        for i in range(len(self.coordinates) - 1):
            lat1, lon1 = self.coordinates[i]
            lat2, lon2 = self.coordinates[i + 1]
            total += grid.distance_km(lat1, lon1, lat2, lon2)
        return total
    
    def get_dominant_segment(self) -> Optional[LineSegment]:
        """Get the segment with highest weight."""
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: s.weight)
    
    def get_weighted_attribute(self,
                              attr_name: str,
                              aggregation: str = 'weighted_mean') -> Optional[float]:
        """Calculate segment-weighted attribute value.
        
        Args:
            attr_name: Attribute name to aggregate
            aggregation: Aggregation method
                - 'weighted_mean': Weight-weighted average
                - 'mean': Simple average across segments
                - 'sum': Sum of values
                - 'max': Maximum value
                - 'min': Minimum value
                - 'median': Median value
                - 'length_weighted': Length-weighted average
        
        Returns:
            Aggregated value or None if attribute not found
        """
        if not self.segments:
            return None
        
        values = []
        weights = []
        
        for seg in self.segments:
            val = seg.get_attribute(attr_name)
            if val is not None:
                try:
                    values.append(float(val))
                    if aggregation == 'length_weighted':
                        weights.append(seg.length_km)
                    else:
                        weights.append(seg.weight)
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return None
        
        if aggregation in ['weighted_mean', 'length_weighted']:
            total_weight = sum(weights)
            if total_weight == 0:
                return statistics.mean(values)
            return sum(v * w for v, w in zip(values, weights)) / total_weight
        elif aggregation == 'mean':
            return statistics.mean(values)
        elif aggregation == 'sum':
            return sum(values)
        elif aggregation == 'max':
            return max(values)
        elif aggregation == 'min':
            return min(values)
        elif aggregation == 'median':
            return statistics.median(values)
        
        return None
    
    def get_attribute(self, attr_name: str, default: Any = None) -> Any:
        """Get polyline-level attribute."""
        return self.attributes.get(attr_name, default)
    
    def set_attribute(self, attr_name: str, value: Any) -> None:
        """Set polyline-level attribute."""
        self.attributes[attr_name] = value
    
    def interpolate_points(self, step_km: float = 0.5) -> List[Tuple[float, float]]:
        """Interpolate points along the polyline at regular intervals.
        
        Args:
            step_km: Distance between interpolated points in kilometers
        
        Returns:
            List of (lat, lon) tuples
        """
        if len(self.coordinates) < 2:
            return list(self.coordinates)
        
        grid = DGGSS2()
        points = []
        
        for i in range(len(self.coordinates) - 1):
            lat1, lon1 = self.coordinates[i]
            lat2, lon2 = self.coordinates[i + 1]
            
            dist = grid.distance_km(lat1, lon1, lat2, lon2)
            n_points = max(1, int(dist / step_km))
            
            for j in range(n_points):
                t = j / n_points
                lat = lat1 + (lat2 - lat1) * t
                lon = lon1 + (lon2 - lon1) * t
                points.append((lat, lon))
        
        # Add final point
        points.append(self.coordinates[-1])
        return points


####################################################################
# Generic Polyline Discretization Functions
####################################################################

def discretize_polyline_features(
    features: Sequence[PolylineFeature],
    level: int = 12,
    method: str = 'interpolate',
    step_km: float = 0.5,
    include_topology: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Discretize polyline features to DGGS cells.
    
    Three discretization methods:
    1. 'interpolate': Sample points along line at regular intervals (default)
    2. 'vertices': Use only line vertices
    3. 'buffer': Cover area around line (not yet implemented)
    
    Args:
        features: List of PolylineFeature objects
        level: Target DGGS cell level (10-15)
        method: 'interpolate', 'vertices', or 'buffer'
        step_km: Interpolation step size in kilometers (for interpolate method)
        include_topology: Include connectivity information
    
    Returns:
        Dictionary mapping cell_token -> {
            'features': List[feature_id],  # Features passing through cell
            'feature_types': Set[str],
            'total_length_km': float,      # Total line length in cell
            'directions': Set[str],        # If directed: 'inbound', 'outbound', 'through'
            'num_features': int
        }
    
    Example:
        features = [
            PolylineFeature('road1', [(40.0, -74.0), (40.1, -74.1)], 
                          feature_type='highway')
        ]
        result = discretize_polyline_features(features, level=12, method='interpolate')
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        # Get points based on method
        if method == 'interpolate':
            points = feature.interpolate_points(step_km=step_km)
        elif method == 'vertices':
            points = list(feature.coordinates)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Map points to cells
        cell_tokens = []
        for lat, lon in points:
            token = grid.latlon_to_token(lat, lon, level)
            cell_tokens.append(token)
        
        # Remove duplicates while preserving order
        unique_tokens = []
        seen = set()
        for token in cell_tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        # Store feature info in each cell
        for token in unique_tokens:
            if token not in result:
                result[token] = {
                    'features': [],
                    'feature_types': set(),
                    'total_length_km': 0.0,
                    'num_features': 0
                }
            
            result[token]['features'].append(feature.feature_id)
            result[token]['feature_types'].add(feature.feature_type)
            result[token]['num_features'] = len(result[token]['features'])
            
            # Estimate length contribution (rough approximation)
            length_per_cell = feature.total_length_km / len(unique_tokens)
            result[token]['total_length_km'] += length_per_cell
    
    # Convert sets to lists for JSON serialization
    for data in result.values():
        data['feature_types'] = list(data['feature_types'])
    
    return result


def discretize_polyline_attributes(
    features: Sequence[PolylineFeature],
    attributes: List[str],
    level: int = 12,
    aggregation_funcs: Optional[Dict[str, str]] = None,
    weight_by_length: bool = True,
    step_km: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """Discretize polyline attributes with statistical aggregation.
    
    Aggregates segment-level attributes to DGGS cells using various
    statistical methods. Supports weighted and unweighted aggregation.
    
    Args:
        features: List of PolylineFeature objects
        attributes: List of attribute names to aggregate
        level: Target DGGS cell level
        aggregation_funcs: Dict mapping attribute -> aggregation function
            Options: 'weighted_mean', 'length_weighted', 'mean', 'sum', 'max', 'min', 'median'
        weight_by_length: If True, weight by segment length
        step_km: Interpolation step for cell mapping
    
    Returns:
        Dictionary mapping cell_token -> {
            'attr1_mean': value,
            'attr2_length_weighted': value,
            ...
        }
    
    Example:
        features = [
            PolylineFeature('road1', coords, segments=[
                LineSegment('seg1', start, end, {'speed_limit': 50, 'lanes': 2}),
                LineSegment('seg2', start, end, {'speed_limit': 65, 'lanes': 4})
            ])
        ]
        
        result = discretize_polyline_attributes(
            features,
            attributes=['speed_limit', 'lanes'],
            aggregation_funcs={'speed_limit': 'length_weighted', 'lanes': 'max'}
        )
    """
    if aggregation_funcs is None:
        aggregation_funcs = {
            attr: 'length_weighted' if weight_by_length else 'mean'
            for attr in attributes
        }
    
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        # Get cells that this feature passes through
        points = feature.interpolate_points(step_km=step_km)
        cell_tokens = set()
        for lat, lon in points:
            token = grid.latlon_to_token(lat, lon, level)
            cell_tokens.add(token)
        
        # Aggregate attributes for each cell
        for token in cell_tokens:
            if token not in result:
                result[token] = {
                    'feature_ids': [],
                    'num_features': 0
                }
            
            result[token]['feature_ids'].append(feature.feature_id)
            result[token]['num_features'] = len(result[token]['feature_ids'])
            
            # Aggregate each attribute
            for attr in attributes:
                agg_func = aggregation_funcs.get(attr, 'length_weighted')
                value = feature.get_weighted_attribute(attr, aggregation=agg_func)
                
                if value is not None:
                    result[token][f'{attr}_{agg_func}'] = value
    
    return result


def discretize_polyline_network(
    features: Sequence[PolylineFeature],
    level: int = 12,
    connectivity_threshold_km: float = 0.1,
    step_km: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """Discretize polyline network with connectivity analysis.
    
    Analyzes network topology and identifies:
    - Network nodes (intersections, confluences)
    - Network edges (segments between nodes)
    - Connectivity patterns
    - Flow directions
    
    Args:
        features: List of PolylineFeature objects
        level: Target DGGS cell level
        connectivity_threshold_km: Distance threshold for considering features connected
        step_km: Interpolation step for cell mapping
    
    Returns:
        Dictionary mapping cell_token -> {
            'is_node': bool,              # True if intersection/junction
            'connected_features': List[str],
            'num_connections': int,
            'node_type': str,             # 'junction', 'endpoint', 'through'
            'incoming': List[str],        # For directed networks
            'outgoing': List[str],        # For directed networks
        }
    
    Example:
        # Road network
        result = discretize_polyline_network(roads, level=13)
        
        # River network
        result = discretize_polyline_network(rivers, level=12)
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    # First pass: Map all features to cells
    feature_cells: Dict[str, Set[str]] = {}  # feature_id -> set of cell tokens
    
    for feature in features:
        points = feature.interpolate_points(step_km=step_km)
        cell_tokens = set()
        
        for lat, lon in points:
            token = grid.latlon_to_token(lat, lon, level)
            cell_tokens.add(token)
        
        feature_cells[feature.feature_id] = cell_tokens
    
    # Second pass: Analyze connectivity
    for token in set().union(*feature_cells.values()):
        # Find features passing through this cell
        features_in_cell = [
            fid for fid, tokens in feature_cells.items()
            if token in tokens
        ]
        
        if token not in result:
            result[token] = {
                'is_node': False,
                'connected_features': features_in_cell,
                'num_connections': len(features_in_cell),
                'node_type': 'through'
            }
        
        # Determine node type
        if len(features_in_cell) > 2:
            result[token]['is_node'] = True
            result[token]['node_type'] = 'junction'
        elif len(features_in_cell) == 1:
            # Check if this is an endpoint
            feature = next((f for f in features if f.feature_id == features_in_cell[0]), None)
            if feature:
                start_token = grid.latlon_to_token(feature.start_point[0], feature.start_point[1], level)
                end_token = grid.latlon_to_token(feature.end_point[0], feature.end_point[1], level)
                
                if token == start_token or token == end_token:
                    result[token]['is_node'] = True
                    result[token]['node_type'] = 'endpoint'
    
    return result


def discretize_polyline_density(
    features: Sequence[PolylineFeature],
    level: int = 12,
    step_km: float = 0.5,
    normalize_by_area: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Calculate polyline density (length per unit area) in each cell.
    
    Useful for:
    - Road density analysis
    - Stream density analysis
    - Infrastructure coverage assessment
    
    Args:
        features: List of PolylineFeature objects
        level: Target DGGS cell level
        step_km: Interpolation step
        normalize_by_area: If True, return km/km² (density), else total km
    
    Returns:
        Dictionary mapping cell_token -> {
            'total_length_km': float,
            'density_km_per_km2': float,  # If normalize_by_area=True
            'feature_count': int,
            'avg_length_per_feature': float
        }
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        points = feature.interpolate_points(step_km=step_km)
        cell_tokens = []
        
        for lat, lon in points:
            token = grid.latlon_to_token(lat, lon, level)
            cell_tokens.append(token)
        
        # Remove duplicates
        unique_tokens = list(set(cell_tokens))
        
        # Distribute length across cells
        length_per_cell = feature.total_length_km / len(unique_tokens) if unique_tokens else 0
        
        for token in unique_tokens:
            if token not in result:
                result[token] = {
                    'total_length_km': 0.0,
                    'feature_count': 0,
                    'features': []
                }
            
            result[token]['total_length_km'] += length_per_cell
            result[token]['features'].append(feature.feature_id)
            result[token]['feature_count'] = len(set(result[token]['features']))
    
    # Calculate density if requested
    if normalize_by_area:
        for token, data in result.items():
            # Get cell area (approximate from S2 cell)
            cell_id = CellId.from_token(token)
            cell = Cell(cell_id)
            # S2 area is in steradians, convert to km²
            # Earth radius = 6371 km, area = steradians * R²
            cell_area_km2 = cell.exact_area() * (6371.0 ** 2)
            data['density_km_per_km2'] = data['total_length_km'] / cell_area_km2
            data['avg_length_per_feature'] = data['total_length_km'] / data['feature_count']
    
    # Clean up features list
    for data in result.values():
        data['features'] = list(set(data['features']))
    
    return result


def discretize_polyline_flow(
    features: Sequence[PolylineFeature],
    flow_attribute: str,
    level: int = 12,
    step_km: float = 0.5,
    accumulate: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Discretize flow-based attributes (e.g., traffic, river discharge).
    
    For directed networks, can accumulate flow downstream/downstream.
    
    Args:
        features: List of PolylineFeature objects
        flow_attribute: Name of flow attribute (e.g., 'discharge', 'traffic_volume')
        level: Target DGGS cell level
        step_km: Interpolation step
        accumulate: If True, accumulate flow along directed paths
    
    Returns:
        Dictionary mapping cell_token -> {
            'total_flow': float,
            'avg_flow': float,
            'max_flow': float,
            'flow_direction': str  # For directed features
        }
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        points = feature.interpolate_points(step_km=step_km)
        
        # Get flow value
        flow_value = feature.get_weighted_attribute(flow_attribute, aggregation='mean')
        if flow_value is None:
            continue
        
        for lat, lon in points:
            token = grid.latlon_to_token(lat, lon, level)
            
            if token not in result:
                result[token] = {
                    'flows': [],
                    'total_flow': 0.0,
                    'feature_ids': []
                }
            
            result[token]['flows'].append(flow_value)
            result[token]['feature_ids'].append(feature.feature_id)
    
    # Calculate statistics
    for token, data in result.items():
        flows = data['flows']
        data['total_flow'] = sum(flows)
        data['avg_flow'] = statistics.mean(flows) if flows else 0
        data['max_flow'] = max(flows) if flows else 0
        data['min_flow'] = min(flows) if flows else 0
        data['num_features'] = len(set(data['feature_ids']))
        
        # Clean up
        del data['flows']
    
    return result


####################################################################
# Utility Functions
####################################################################

def create_polyline_feature_from_dict(data: Dict[str, Any]) -> PolylineFeature:
    """Create PolylineFeature from dictionary.
    
    Useful for loading from JSON, CSV, or databases.
    
    Args:
        data: Dictionary with keys:
            - 'feature_id': str
            - 'coordinates': List[(lat, lon)]
            - 'segments': List[dict] (optional)
            - 'attributes': dict (optional)
            - 'feature_type': str (optional)
            - 'is_directed': bool (optional)
    
    Returns:
        PolylineFeature object
    """
    segments = []
    for seg_data in data.get('segments', []):
        seg = LineSegment(
            segment_id=seg_data.get('segment_id', ''),
            start_point=tuple(seg_data.get('start_point', (0, 0))),
            end_point=tuple(seg_data.get('end_point', (0, 0))),
            attributes=seg_data.get('attributes', {}),
            weight=seg_data.get('weight', 1.0)
        )
        segments.append(seg)
    
    return PolylineFeature(
        feature_id=data['feature_id'],
        coordinates=data['coordinates'],
        segments=segments,
        attributes=data.get('attributes', {}),
        feature_type=data.get('feature_type'),
        is_directed=data.get('is_directed', False)
    )


def calculate_polyline_sinuosity(feature: PolylineFeature) -> float:
    """Calculate sinuosity index (actual length / straight-line distance).
    
    Sinuosity = 1.0 for straight lines, >1.0 for curved/meandering lines.
    
    Args:
        feature: PolylineFeature object
    
    Returns:
        Sinuosity index
    """
    if len(feature.coordinates) < 2:
        return 1.0
    
    grid = DGGSS2()
    
    # Straight-line distance
    start = feature.start_point
    end = feature.end_point
    straight_distance = grid.distance_km(start[0], start[1], end[0], end[1])
    
    if straight_distance == 0:
        return 1.0
    
    # Actual length
    actual_length = feature.total_length_km
    
    return actual_length / straight_distance


def merge_polyline_networks(
    networks: List[Dict[str, Dict[str, Any]]],
    merge_strategy: str = 'union'
) -> Dict[str, Dict[str, Any]]:
    """Merge multiple polyline network discretizations.
    
    Args:
        networks: List of discretization results
        merge_strategy: 'union' or 'intersection'
    
    Returns:
        Merged network dictionary
    """
    if not networks:
        return {}
    
    if merge_strategy == 'union':
        result = {}
        for network in networks:
            for token, data in network.items():
                if token not in result:
                    result[token] = data.copy()
                else:
                    # Merge data
                    if 'total_length_km' in data:
                        result[token]['total_length_km'] = result[token].get('total_length_km', 0) + data['total_length_km']
                    if 'features' in data:
                        result[token]['features'] = list(set(result[token].get('features', []) + data['features']))
        return result
    
    elif merge_strategy == 'intersection':
        # Find common cells
        all_tokens = [set(net.keys()) for net in networks]
        common_tokens = set.intersection(*all_tokens)
        
        result = {}
        for token in common_tokens:
            result[token] = networks[0][token].copy()
        return result
    
    return {}
