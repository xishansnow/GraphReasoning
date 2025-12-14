"""
DGGS Point Discretization Module

Generic discretization framework for point vector data with attributes.
Supports any point-based spatial data (weather stations, POIs, sampling sites, etc.)
with attributes, clusters, and hierarchical structure.

Key concepts:
- Point: Geographic location defined by coordinates
- Attributes: Properties associated with points
- Clusters: Groups of related points
- Spatial patterns: Density, distribution, clustering

Supported data types:
- Environmental monitoring (weather stations, air quality sensors)
- Points of Interest (POIs) (restaurants, hotels, landmarks)
- Sampling sites (soil samples, water quality, biodiversity)
- Infrastructure (cell towers, fire hydrants, street lights)
- Events (crime incidents, accidents, disease cases)
- Facilities (hospitals, schools, fire stations)
- Natural features (trees, springs, peaks)

Architecture:
    PointFeature (base class)
        ├── Attributes (properties at location)
        └── Discretization methods (direct, cluster, density, pattern)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, Set
from s2sphere import CellId, LatLng, Cell
from .dggs import DggsS2
from .spatial_utils import discretize_aggregate
import statistics
import math


####################################################################
# Generic Point Data Models
####################################################################

class PointFeature:
    """Generic point feature with attributes.
    
    Represents any point-based spatial feature with:
    - Geographic coordinates (single location)
    - Attributes (properties/measurements)
    - Optional cluster/group membership
    - Temporal information (optional)
    
    This is the base class for domain-specific implementations like
    weather stations, POIs, sampling sites, etc.
    """
    
    def __init__(self,
                 feature_id: str,
                 latitude: float,
                 longitude: float,
                 attributes: Optional[Dict[str, Any]] = None,
                 feature_type: Optional[str] = None,
                 cluster_id: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 weight: float = 1.0):
        """
        Args:
            feature_id: Unique identifier for this point
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            attributes: Point attributes (measurements, properties)
            feature_type: Type descriptor (e.g., 'weather_station', 'poi', 'sample')
            cluster_id: Optional cluster/group identifier
            timestamp: Optional timestamp (ISO format or custom)
            weight: Relative importance/weight (for weighted aggregations)
        """
        self.feature_id = feature_id
        self.latitude = latitude
        self.longitude = longitude
        self.attributes = attributes or {}
        self.feature_type = feature_type or 'generic_point'
        self.cluster_id = cluster_id
        self.timestamp = timestamp
        self.weight = weight
        self.coordinates = (latitude, longitude)
    
    def get_attribute(self, attr_name: str, default: Any = None) -> Any:
        """Get attribute value with fallback."""
        return self.attributes.get(attr_name, default)
    
    def set_attribute(self, attr_name: str, value: Any) -> None:
        """Set attribute value."""
        self.attributes[attr_name] = value
    
    def distance_to(self, other: 'PointFeature', grid: Optional[DGGSS2] = None) -> float:
        """Calculate distance to another point in kilometers.
        
        Args:
            other: Another PointFeature
            grid: Optional DGGSS2 instance (creates new if not provided)
        
        Returns:
            Distance in kilometers
        """
        if grid is None:
            grid = DGGSS2()
        return grid.distance_km(
            self.latitude, self.longitude,
            other.latitude, other.longitude
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_id': self.feature_id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'attributes': self.attributes,
            'feature_type': self.feature_type,
            'cluster_id': self.cluster_id,
            'timestamp': self.timestamp,
            'weight': self.weight
        }


class PointCluster:
    """Represents a cluster/group of related points.
    
    Used for grouping points by:
    - Spatial proximity
    - Shared attributes
    - Functional relationships
    - Administrative groupings
    """
    
    def __init__(self,
                 cluster_id: str,
                 points: List[PointFeature],
                 cluster_type: Optional[str] = None,
                 attributes: Optional[Dict[str, Any]] = None):
        """
        Args:
            cluster_id: Unique cluster identifier
            points: List of PointFeature objects in this cluster
            cluster_type: Type of cluster (spatial, functional, etc.)
            attributes: Cluster-level attributes
        """
        self.cluster_id = cluster_id
        self.points = points
        self.cluster_type = cluster_type or 'generic'
        self.attributes = attributes or {}
        self.centroid = self._calculate_centroid()
        self.num_points = len(points)
    
    def _calculate_centroid(self) -> Tuple[float, float]:
        """Calculate cluster centroid (average coordinates)."""
        if not self.points:
            return (0.0, 0.0)
        
        avg_lat = statistics.mean(p.latitude for p in self.points)
        avg_lon = statistics.mean(p.longitude for p in self.points)
        return (avg_lat, avg_lon)
    
    def get_aggregate_attribute(self,
                                attr_name: str,
                                aggregation: str = 'mean') -> Optional[float]:
        """Aggregate attribute across cluster points.
        
        Args:
            attr_name: Attribute name to aggregate
            aggregation: 'mean', 'sum', 'max', 'min', 'median', 'weighted_mean'
        
        Returns:
            Aggregated value or None if attribute not found
        """
        values = []
        weights = []
        
        for point in self.points:
            val = point.get_attribute(attr_name)
            if val is not None:
                try:
                    values.append(float(val))
                    weights.append(point.weight)
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return None
        
        if aggregation == 'weighted_mean':
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


####################################################################
# Generic Point Discretization Functions
####################################################################

def discretize_point_features(
    features: Sequence[PointFeature],
    level: int = 12,
    include_coordinates: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Discretize all point features into DGGS cells with direct mapping.   
    Direct mapping: Each point maps to exactly one cell at the target level.
    Multiple points can map to the same cell.
    
    Args:
        features: List of PointFeature objects
        level: Target DGGS cell level (10-15)
        include_coordinates: Include original point coordinates in result
    
    Returns:
        Dictionary mapping cell_token -> {
            'features': List[feature_id],
            'feature_types': Set[str],
            'num_points': int,
            'coordinates': List[(lat, lon)] (if include_coordinates=True)
        }
    
    Example:
        features = [
            PointFeature('station1', 40.0, -74.0, 
                        attributes={'temp': 25}, feature_type='weather'),
            PointFeature('station2', 40.01, -74.01,
                        attributes={'temp': 26}, feature_type='weather')
        ]
        result = discretize_point_features(features, level=12)
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        token = grid.latlon_to_token(feature.latitude, feature.longitude, level)
        
        if token not in result:
            result[token] = {
                'features': [],
                'feature_types': set(),
                'num_points': 0
            }
            if include_coordinates:
                result[token]['coordinates'] = []
        
        result[token]['features'].append(feature.feature_id)
        result[token]['feature_types'].add(feature.feature_type)
        result[token]['num_points'] = len(result[token]['features'])
        
        if include_coordinates:
            result[token]['coordinates'].append((feature.latitude, feature.longitude))
    
    # Convert sets to lists for JSON serialization
    for data in result.values():
        data['feature_types'] = list(data['feature_types'])
    
    return result


def discretize_point_attributes(
    features: Sequence[PointFeature],
    attributes: List[str],
    level: int = 12,
    aggregation_funcs: Optional[Dict[str, str]] = None,
    weight_by: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Discretize point attributes with statistical aggregation in each DGGS cell.
    
    Aggregates attributes from points in the same DGGS cell using various methods.
    Supports weighted and unweighted aggregation.
    
    Args:
        features: List of PointFeature objects
        attributes: List of attribute names to aggregate
        level: Target DGGS cell level
        aggregation_funcs: Dict mapping attribute -> aggregation function
            Options: 'mean', 'sum', 'max', 'min', 'median', 'weighted_mean', 'count', 'std'
        weight_by: Attribute name to use for weighting (if None, use feature.weight)
    
    Returns:
        Dictionary mapping cell_token -> {
            'attr1_mean': value,
            'attr2_max': value,
            'num_points': int,
            'point_ids': List[str]
        }
    
    Example:
        features = [
            PointFeature('s1', 40.0, -74.0, attributes={'temp': 25, 'humidity': 60}),
            PointFeature('s2', 40.01, -74.01, attributes={'temp': 26, 'humidity': 65})
        ]
        
        result = discretize_point_attributes(
            features,
            attributes=['temp', 'humidity'],
            aggregation_funcs={'temp': 'mean', 'humidity': 'max'}
        )
    """
    if aggregation_funcs is None:
        aggregation_funcs = {attr: 'mean' for attr in attributes}
    
    grid = DGGSS2(level=level)
    
    # Group points by cell
    cell_points: Dict[str, List[PointFeature]] = {}
    for feature in features:
        token = grid.latlon_to_token(feature.latitude, feature.longitude, level)
        if token not in cell_points:
            cell_points[token] = []
        cell_points[token].append(feature)
    
    # Aggregate attributes for each cell
    result: Dict[str, Dict[str, Any]] = {}
    
    for token, points in cell_points.items():
        result[token] = {
            'num_points': len(points),
            'point_ids': [p.feature_id for p in points]
        }
        
        # Aggregate each attribute
        for attr in attributes:
            agg_func = aggregation_funcs.get(attr, 'mean')
            
            # Collect values and weights
            values = []
            weights = []
            
            for point in points:
                val = point.get_attribute(attr)
                if val is not None:
                    try:
                        values.append(float(val))
                        if weight_by:
                            weight = point.get_attribute(weight_by, 1.0)
                            weights.append(float(weight))
                        else:
                            weights.append(point.weight)
                    except (ValueError, TypeError):
                        continue
            
            if not values:
                continue
            
            # Calculate aggregation
            if agg_func == 'weighted_mean':
                total_weight = sum(weights)
                if total_weight == 0:
                    agg_value = statistics.mean(values)
                else:
                    agg_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            elif agg_func == 'mean':
                agg_value = statistics.mean(values)
            elif agg_func == 'sum':
                agg_value = sum(values)
            elif agg_func == 'max':
                agg_value = max(values)
            elif agg_func == 'min':
                agg_value = min(values)
            elif agg_func == 'median':
                agg_value = statistics.median(values)
            elif agg_func == 'count':
                agg_value = len(values)
            elif agg_func == 'std':
                agg_value = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                agg_value = statistics.mean(values)
            
            result[token][f'{attr}_{agg_func}'] = agg_value
    
    return result


def discretize_point_density(
    features: Sequence[PointFeature],
    level: int = 12,
    normalize_by_area: bool = True,
    kernel_radius_cells: int = 0
) -> Dict[str, Dict[str, Any]]:
    """Calculate point density in each cell.
    
    Computes the number of points per unit area, optionally using
    kernel density estimation with neighboring cells.
    
    Args:
        features: List of PointFeature objects
        level: Target DGGS cell level
        normalize_by_area: If True, return density (points/km²), else count
        kernel_radius_cells: If > 0, include neighbor cells in density (KDE)
    
    Returns:
        Dictionary mapping cell_token -> {
            'point_count': int,
            'density_points_per_km2': float (if normalize_by_area=True),
            'feature_ids': List[str]
        }
    
    Example:
        # Simple density
        density = discretize_point_density(weather_stations, level=12)
        
        # Kernel density (includes neighbors)
        kde = discretize_point_density(crime_points, level=13, kernel_radius_cells=1)
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    # Count points per cell
    for feature in features:
        token = grid.latlon_to_token(feature.latitude, feature.longitude, level)
        
        if token not in result:
            result[token] = {
                'point_count': 0,
                'feature_ids': []
            }
        
        result[token]['point_count'] += 1
        result[token]['feature_ids'].append(feature.feature_id)
    
    # Apply kernel density if requested
    if kernel_radius_cells > 0:
        # Get neighbor cells for each cell
        kernel_result = {}
        for token in result.keys():
            # Get neighbors using the grid.neighbors method
            neighbors = grid.neighbors(token, ring=kernel_radius_cells)
            
            # Sum counts from this cell and neighbors
            kernel_count = result[token]['point_count']
            for neighbor_token in neighbors:
                if neighbor_token in result:
                    kernel_count += result[neighbor_token]['point_count']
            
            kernel_result[token] = {
                'point_count': result[token]['point_count'],
                'kernel_count': kernel_count,
                'feature_ids': result[token]['feature_ids']
            }
        result = kernel_result
    
    # Normalize by area if requested
    if normalize_by_area:
        for token, data in result.items():
            cell_id = CellId.from_token(token)
            cell = Cell(cell_id)
            cell_area_km2 = cell.exact_area() * (6371.0 ** 2)
            
            count_key = 'kernel_count' if kernel_radius_cells > 0 else 'point_count'
            data['density_points_per_km2'] = data[count_key] / cell_area_km2
    
    return result


def discretize_point_clusters(
    features: Sequence[PointFeature],
    level: int = 12,
    cluster_by: str = 'cluster_id',
    aggregate_attributes: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Discretize point clusters with cluster-level statistics in each DGGS cell.
    
    Groups points by cluster membership and aggregates within clusters.
    Useful for analyzing grouped phenomena (e.g., crime hotspots, disease clusters).
    
    Args:
        features: List of PointFeature objects
        level: Target DGGS cell level
        cluster_by: Attribute to use for clustering ('cluster_id' by default)
        aggregate_attributes: Optional list of attributes to aggregate per cluster
    
    Returns:
        Dictionary mapping cell_token -> {
            'clusters': Dict[cluster_id -> num_points],
            'num_clusters': int,
            'total_points': int,
            'dominant_cluster': str (cluster with most points),
            'cluster_diversity': float (Shannon entropy)
        }
    
    Example:
        # Points with cluster IDs
        points = [
            PointFeature('p1', 40.0, -74.0, cluster_id='cluster_A'),
            PointFeature('p2', 40.01, -74.01, cluster_id='cluster_A'),
            PointFeature('p3', 40.02, -74.02, cluster_id='cluster_B')
        ]
        
        result = discretize_point_clusters(points, level=12)       
        
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        token = grid.latlon_to_token(feature.latitude, feature.longitude, level)
        
        # Get cluster identifier
        if cluster_by == 'cluster_id':
            cluster = feature.cluster_id or 'unclustered'
        else:
            cluster = feature.get_attribute(cluster_by, 'unclustered')
        
        if token not in result:
            result[token] = {
                'clusters': {},
                'points_by_cluster': {},
                'total_points': 0
            }
        
        # Count points per cluster
        if cluster not in result[token]['clusters']:
            result[token]['clusters'][cluster] = 0
            result[token]['points_by_cluster'][cluster] = []
        
        result[token]['clusters'][cluster] += 1
        result[token]['points_by_cluster'][cluster].append(feature)
        result[token]['total_points'] += 1
    
    # Calculate cluster statistics
    for token, data in result.items():
        data['num_clusters'] = len(data['clusters'])
        
        # Dominant cluster
        if data['clusters']:
            data['dominant_cluster'] = max(data['clusters'], key=data['clusters'].get)
            
            # Cluster diversity (Shannon entropy)
            total = data['total_points']
            entropy = 0.0
            for count in data['clusters'].values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            data['cluster_diversity'] = entropy
        
        # Aggregate attributes per cluster if requested
        if aggregate_attributes:
            data['cluster_attributes'] = {}
            for cluster_id, points in data['points_by_cluster'].items():
                data['cluster_attributes'][cluster_id] = {}
                
                for attr in aggregate_attributes:
                    values = []
                    for point in points:
                        val = point.get_attribute(attr)
                        if val is not None:
                            try:
                                values.append(float(val))
                            except (ValueError, TypeError):
                                continue
                    
                    if values:
                        data['cluster_attributes'][cluster_id][attr] = {
                            'mean': statistics.mean(values),
                            'min': min(values),
                            'max': max(values)
                        }
        
        # Clean up intermediate data
        del data['points_by_cluster']
    
    return result


def discretize_point_patterns(
    features: Sequence[PointFeature],
    level: int = 12,
    pattern_metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Analyze spatial patterns of points in each cell.
    
    Calculates various spatial statistics to characterize point distributions:
    - Centroid: Average location
    - Spread: Standard distance from centroid
    - Dispersion: Coefficient of variation
    - Nearest neighbor: Average distance to nearest point
    
    Args:
        features: List of PointFeature objects
        level: Target DGGS cell level
        pattern_metrics: List of metrics to compute
            Options: 'centroid', 'spread', 'dispersion', 'nearest_neighbor', 'all'
    
    Returns:
        Dictionary mapping cell_token -> {
            'centroid': (lat, lon),
            'spread_km': float,
            'dispersion': float,
            'avg_nearest_neighbor_km': float,
            'num_points': int
        }
    
    Example:
        # Analyze tree distribution patterns
        patterns = discretize_point_patterns(
            tree_locations,
            level=13,
            pattern_metrics=['centroid', 'spread', 'nearest_neighbor']
        )
    """
    if pattern_metrics is None or 'all' in pattern_metrics:
        pattern_metrics = ['centroid', 'spread', 'dispersion', 'nearest_neighbor']
    
    grid = DGGSS2(level=level)
    
    # Group points by cell
    cell_points: Dict[str, List[PointFeature]] = {}
    for feature in features:
        token = grid.latlon_to_token(feature.latitude, feature.longitude, level)
        if token not in cell_points:
            cell_points[token] = []
        cell_points[token].append(feature)
    
    result: Dict[str, Dict[str, Any]] = {}
    
    for token, points in cell_points.items():
        result[token] = {'num_points': len(points)}
        
        if len(points) < 2:
            # Need at least 2 points for meaningful statistics
            if 'centroid' in pattern_metrics:
                result[token]['centroid'] = (points[0].latitude, points[0].longitude)
            continue
        
        # Calculate centroid
        if 'centroid' in pattern_metrics:
            centroid_lat = statistics.mean(p.latitude for p in points)
            centroid_lon = statistics.mean(p.longitude for p in points)
            result[token]['centroid'] = (centroid_lat, centroid_lon)
        else:
            centroid_lat = statistics.mean(p.latitude for p in points)
            centroid_lon = statistics.mean(p.longitude for p in points)
        
        # Calculate spread (standard distance from centroid)
        if 'spread' in pattern_metrics or 'dispersion' in pattern_metrics:
            distances = []
            for point in points:
                dist = grid.distance_km(
                    point.latitude, point.longitude,
                    centroid_lat, centroid_lon
                )
                distances.append(dist)
            
            if 'spread' in pattern_metrics:
                result[token]['spread_km'] = statistics.stdev(distances) if len(distances) > 1 else 0.0
            
            if 'dispersion' in pattern_metrics:
                mean_dist = statistics.mean(distances)
                if mean_dist > 0:
                    cv = statistics.stdev(distances) / mean_dist if len(distances) > 1 else 0.0
                    result[token]['dispersion'] = cv
                else:
                    result[token]['dispersion'] = 0.0
        
        # Calculate nearest neighbor distances
        if 'nearest_neighbor' in pattern_metrics:
            nn_distances = []
            for i, point in enumerate(points):
                min_dist = float('inf')
                for j, other in enumerate(points):
                    if i != j:
                        dist = grid.distance_km(
                            point.latitude, point.longitude,
                            other.latitude, other.longitude
                        )
                        min_dist = min(min_dist, dist)
                if min_dist != float('inf'):
                    nn_distances.append(min_dist)
            
            if nn_distances:
                result[token]['avg_nearest_neighbor_km'] = statistics.mean(nn_distances)
    
    return result


def discretize_point_temporal(
    features: Sequence[PointFeature],
    level: int = 12,
    time_attribute: str = 'timestamp',
    temporal_aggregation: str = 'count'
) -> Dict[str, Dict[str, Any]]:
    """Analyzes temporal patterns of point data in each cell.
    Useful for event data with timestamps.
    
    Args:
        features: List of PointFeature objects
        level: Target DGGS cell level
        time_attribute: Attribute name containing timestamp
        temporal_aggregation: How to aggregate temporal data
            Options: 'count', 'first', 'last', 'duration', 'frequency'
    
    Returns:
        Dictionary mapping cell_token -> {
            'total_events': int,
            'first_event': str (timestamp),
            'last_event': str (timestamp),
            'event_ids': List[str]
        }
    
    Example:
        # Crime incidents with timestamps
        temporal = discretize_point_temporal(
            crime_points,
            level=13,
            time_attribute='timestamp'
        )
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        token = grid.latlon_to_token(feature.latitude, feature.longitude, level)
        
        if token not in result:
            result[token] = {
                'total_events': 0,
                'event_ids': [],
                'timestamps': []
            }
        
        result[token]['total_events'] += 1
        result[token]['event_ids'].append(feature.feature_id)
        
        # Get timestamp
        if time_attribute == 'timestamp':
            timestamp = feature.timestamp
        else:
            timestamp = feature.get_attribute(time_attribute)
        
        if timestamp:
            result[token]['timestamps'].append(timestamp)
    
    # Temporal aggregation
    for token, data in result.items():
        if data['timestamps']:
            # Sort timestamps
            sorted_times = sorted(data['timestamps'])
            data['first_event'] = sorted_times[0]
            data['last_event'] = sorted_times[-1]
            
            # Calculate frequency (events per time unit would need parsing)
            data['num_unique_times'] = len(set(data['timestamps']))
        
        # Clean up
        del data['timestamps']
    
    return result


####################################################################
# Utility Functions
####################################################################

def create_point_feature_from_dict(data: Dict[str, Any]) -> PointFeature:
    """Create PointFeature from dictionary.
    
    Useful for loading from JSON, CSV, or databases.
    
    Args:
        data: Dictionary with keys:
            - 'feature_id': str
            - 'latitude': float
            - 'longitude': float
            - 'attributes': dict (optional)
            - 'feature_type': str (optional)
            - 'cluster_id': str (optional)
            - 'timestamp': str (optional)
            - 'weight': float (optional)
    
    Returns:
        PointFeature object
    """
    return PointFeature(
        feature_id=data['feature_id'],
        latitude=data['latitude'],
        longitude=data['longitude'],
        attributes=data.get('attributes', {}),
        feature_type=data.get('feature_type'),
        cluster_id=data.get('cluster_id'),
        timestamp=data.get('timestamp'),
        weight=data.get('weight', 1.0)
    )


def create_point_clusters_from_features(
    features: Sequence[PointFeature],
    cluster_attribute: str = 'cluster_id'
) -> List[PointCluster]:
    """Group points into clusters based on cluster attribute.
    
    Args:
        features: List of PointFeature objects
        cluster_attribute: Attribute to use for clustering
    
    Returns:
        List of PointCluster objects
    """
    clusters_dict: Dict[str, List[PointFeature]] = {}
    
    for feature in features:
        if cluster_attribute == 'cluster_id':
            cluster_id = feature.cluster_id or 'unclustered'
        else:
            cluster_id = str(feature.get_attribute(cluster_attribute, 'unclustered'))
        
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(feature)
    
    clusters = []
    for cluster_id, points in clusters_dict.items():
        cluster = PointCluster(
            cluster_id=cluster_id,
            points=points,
            cluster_type='attribute_based'
        )
        clusters.append(cluster)
    
    return clusters


def spatial_cluster_points(
    features: Sequence[PointFeature],
    distance_threshold_km: float,
    min_cluster_size: int = 2
) -> List[PointCluster]:
    """Cluster points based on spatial proximity.
    
    Simple distance-based clustering (similar to DBSCAN).
    
    Args:
        features: List of PointFeature objects
        distance_threshold_km: Maximum distance for points to be in same cluster
        min_cluster_size: Minimum number of points to form a cluster
    
    Returns:
        List of PointCluster objects
    """
    grid = DGGSS2()
    clustered = set()
    clusters = []
    cluster_counter = 0
    
    for i, feature in enumerate(features):
        if feature.feature_id in clustered:
            continue
        
        # Find all points within distance threshold
        cluster_points = [feature]
        clustered.add(feature.feature_id)
        
        for j, other in enumerate(features):
            if i != j and other.feature_id not in clustered:
                dist = grid.distance_km(
                    feature.latitude, feature.longitude,
                    other.latitude, other.longitude
                )
                if dist <= distance_threshold_km:
                    cluster_points.append(other)
                    clustered.add(other.feature_id)
        
        # Create cluster if meets minimum size
        if len(cluster_points) >= min_cluster_size:
            cluster = PointCluster(
                cluster_id=f'spatial_cluster_{cluster_counter}',
                points=cluster_points,
                cluster_type='spatial'
            )
            clusters.append(cluster)
            cluster_counter += 1
    
    return clusters


def calculate_point_statistics(
    features: Sequence[PointFeature],
    attribute: str
) -> Dict[str, float]:
    """Calculate global statistics for a point attribute.
    
    Args:
        features: List of PointFeature objects
        attribute: Attribute name to analyze
    
    Returns:
        Dictionary with statistics: mean, std, min, max, median, count
    """
    values = []
    for feature in features:
        val = feature.get_attribute(attribute)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue
    
    if not values:
        return {}
    
    stats = {
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }
    
    if len(values) > 1:
        stats['std'] = statistics.stdev(values)
        stats['median'] = statistics.median(values)
    else:
        stats['std'] = 0.0
        stats['median'] = values[0]
    
    return stats


def filter_points_by_attribute(
    features: Sequence[PointFeature],
    attribute: str,
    condition: Callable[[Any], bool]
) -> List[PointFeature]:
    """Filter points based on attribute condition.
    
    Args:
        features: List of PointFeature objects
        attribute: Attribute name to filter on
        condition: Function that takes attribute value and returns True/False
    
    Returns:
        Filtered list of PointFeature objects
    
    Example:
        # Filter weather stations with temp > 30
        hot_stations = filter_points_by_attribute(
            stations,
            'temperature',
            lambda t: t > 30
        )
    """
    filtered = []
    for feature in features:
        val = feature.get_attribute(attribute)
        if val is not None and condition(val):
            filtered.append(feature)
    return filtered


def merge_point_discretizations(
    discretizations: List[Dict[str, Dict[str, Any]]],
    merge_strategy: str = 'union'
) -> Dict[str, Dict[str, Any]]:
    """Merge multiple point discretization results.
    
    Args:
        discretizations: List of discretization dictionaries
        merge_strategy: 'union' or 'intersection'
    
    Returns:
        Merged discretization dictionary
    """
    if not discretizations:
        return {}
    
    if merge_strategy == 'union':
        result = {}
        for disc in discretizations:
            for token, data in disc.items():
                if token not in result:
                    result[token] = data.copy()
                else:
                    # Merge data (simple concatenation)
                    if 'num_points' in data:
                        result[token]['num_points'] = result[token].get('num_points', 0) + data['num_points']
                    if 'feature_ids' in data:
                        result[token]['feature_ids'] = result[token].get('feature_ids', []) + data['feature_ids']
        return result
    
    elif merge_strategy == 'intersection':
        # Find common cells
        all_tokens = [set(disc.keys()) for disc in discretizations]
        common_tokens = set.intersection(*all_tokens)
        
        result = {}
        for token in common_tokens:
            result[token] = discretizations[0][token].copy()
        return result
    
    return {}
