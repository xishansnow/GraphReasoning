# Point Discretization Module - Quick Reference

Fast reference for discretizing point-based spatial data (weather stations, POIs, sensors, events, facilities, etc.)

---

## 🚀 Quick Start

```python
from DGGS.discretizer_point import (
    PointFeature,
    discretize_point_features,
    discretize_point_attributes
)

# 1. Create points
weather_station = PointFeature(
    feature_id='station_001',
    latitude=40.0,
    longitude=-74.0,
    attributes={'temperature': 25.5, 'humidity': 65},
    feature_type='weather_station'
)

# 2. Discretize to DGGS cells
cells = discretize_point_features([weather_station], level=12)

# 3. Aggregate attributes
attrs = discretize_point_attributes(
    [weather_station],
    attributes=['temperature', 'humidity'],
    aggregation_funcs={'temperature': 'mean', 'humidity': 'mean'}
)
```

---

## 📊 Core Data Models

### PointFeature
```python
point = PointFeature(
    feature_id='point_001',
    latitude=40.0,
    longitude=-74.0,
    attributes={'attr1': value1, 'attr2': value2},
    feature_type='weather_station',  # Type descriptor
    cluster_id='cluster_A',          # Optional cluster membership
    timestamp='2024-01-15T14:30:00', # Optional timestamp
    weight=1.0                       # Relative importance (for weighted aggregations)
)

# Methods
point.get_attribute('temperature')
point.set_attribute('temperature', 25.5)
point.distance_to(other_point)
point.to_dict()  # For serialization
```

### PointCluster
```python
cluster = PointCluster(
    cluster_id='hotspot_1',
    points=[point1, point2, point3],
    cluster_type='spatial',
    attributes={'cluster_attr': value}
)

# Properties
cluster.centroid          # (lat, lon) tuple
cluster.num_points        # Number of points in cluster
cluster.get_aggregate_attribute('temperature', 'mean')
```

---

## 🔧 Core Functions

### 1. Basic Point Discretization
```python
discretize_point_features(
    features=[point1, point2],
    level=12,                    # DGGS level (10-15)
    include_coordinates=False    # Include original coords in result
)
# Returns: {cell_token: {features, feature_types, num_points, [coordinates]}}
```

**Use cases**: Weather stations, POIs, facilities, sampling sites

### 2. Attribute Aggregation
```python
discretize_point_attributes(
    features=[point1, point2],
    attributes=['temp', 'humidity'],
    level=12,
    aggregation_funcs={
        'temp': 'weighted_mean',   # Weight-weighted average
        'humidity': 'mean'          # Simple average
    },
    weight_by='quality'  # Optional: use this attribute for weights
)
# Returns: {cell_token: {temp_weighted_mean, humidity_mean, num_points, point_ids}}
```

**Aggregation methods**: `mean`, `weighted_mean`, `sum`, `max`, `min`, `median`, `count`, `std`

### 3. Point Density
```python
discretize_point_density(
    features=[points],
    level=12,
    normalize_by_area=True,     # Return density (points/km²) vs count
    kernel_radius_cells=0       # 0=simple density, >0=kernel density estimation
)
# Returns: {cell_token: {point_count, density_points_per_km2, feature_ids}}
```

**Use cases**: Crime hotspots, tree density, infrastructure coverage

### 4. Cluster Analysis
```python
discretize_point_clusters(
    features=[points],
    level=12,
    cluster_by='cluster_id',    # Attribute for clustering
    aggregate_attributes=['temp', 'humidity']
)
# Returns: {cell_token: {clusters, num_clusters, total_points, dominant_cluster, 
#                        cluster_diversity, cluster_attributes}}
```

**Use cases**: Crime hotspots, disease clusters, customer segments

### 5. Spatial Patterns
```python
discretize_point_patterns(
    features=[points],
    level=12,
    pattern_metrics=['centroid', 'spread', 'nearest_neighbor']
    # Options: 'centroid', 'spread', 'dispersion', 'nearest_neighbor', 'all'
)
# Returns: {cell_token: {centroid, spread_km, dispersion, avg_nearest_neighbor_km}}
```

**Use cases**: Tree distribution, sampling pattern analysis, point pattern analysis

### 6. Temporal Analysis
```python
discretize_point_temporal(
    features=[events],
    level=12,
    time_attribute='timestamp',
    temporal_aggregation='count'
)
# Returns: {cell_token: {total_events, first_event, last_event, event_ids}}
```

**Use cases**: Crime incidents, disease outbreaks, event tracking

---

## 🎯 Common Use Cases

### Weather Stations
```python
stations = [
    PointFeature('s1', 40.0, -74.0, 
                attributes={'temp': 25, 'humidity': 65, 'pressure': 1013},
                feature_type='weather_station')
]

# Aggregate meteorological data
weather = discretize_point_attributes(
    stations,
    attributes=['temp', 'humidity', 'pressure'],
    aggregation_funcs={
        'temp': 'weighted_mean',
        'humidity': 'mean',
        'pressure': 'mean'
    }
)
```

### Points of Interest (POIs)
```python
restaurants = [
    PointFeature('r1', 40.75, -73.99,
                attributes={'rating': 4.5, 'price_level': 3},
                feature_type='restaurant',
                cluster_id='downtown')
]

# Analyze restaurant distribution and quality
poi_attrs = discretize_point_attributes(
    restaurants,
    attributes=['rating', 'price_level'],
    aggregation_funcs={'rating': 'mean', 'price_level': 'median'}
)

# Neighborhood clustering
clusters = discretize_point_clusters(
    restaurants,
    cluster_by='cluster_id',
    aggregate_attributes=['rating']
)
```

### Crime Incidents
```python
crimes = [
    PointFeature('c1', 40.7, -73.9,
                attributes={'severity': 3},
                feature_type='crime',
                timestamp='2024-01-15T14:30:00',
                cluster_id='hotspot_1')
]

# Hotspot analysis
hotspots = discretize_point_clusters(crimes, cluster_by='cluster_id')

# Temporal patterns
temporal = discretize_point_temporal(crimes, time_attribute='timestamp')

# Density with kernel smoothing
kde = discretize_point_density(
    crimes,
    normalize_by_area=True,
    kernel_radius_cells=1  # Smooth over neighbors
)
```

### Soil Sampling
```python
samples = [
    PointFeature('s1', 39.5, -75.5,
                attributes={'pH': 6.5, 'nitrogen': 25, 'phosphorus': 15},
                feature_type='soil_sample',
                cluster_id='field_A')
]

# Aggregate soil properties
soil = discretize_point_attributes(
    samples,
    attributes=['pH', 'nitrogen', 'phosphorus'],
    aggregation_funcs={
        'pH': 'mean',
        'nitrogen': 'mean',
        'phosphorus': 'mean'
    }
)

# Field-based analysis
fields = discretize_point_clusters(
    samples,
    cluster_by='cluster_id',
    aggregate_attributes=['pH', 'nitrogen']
)

# Sampling patterns
patterns = discretize_point_patterns(
    samples,
    pattern_metrics=['centroid', 'spread', 'nearest_neighbor']
)
```

### Infrastructure (Cell Towers)
```python
towers = [
    PointFeature('t1', 41.0, -74.5,
                attributes={'frequency': 2100, 'power': 50},
                feature_type='cell_tower')
]

# Coverage density with kernel smoothing
density = discretize_point_density(
    towers,
    normalize_by_area=True,
    kernel_radius_cells=1  # Include neighbors for smoother coverage
)

# Signal metrics
coverage = discretize_point_attributes(
    towers,
    attributes=['frequency', 'power'],
    aggregation_funcs={'frequency': 'mean', 'power': 'max'}
)
```

### Air Quality Sensors
```python
sensors = [
    PointFeature('aqi1', 34.05, -118.25,
                attributes={'pm25': 35, 'pm10': 50, 'ozone': 45},
                feature_type='air_quality_sensor',
                weight=1.2)  # Higher quality sensor
]

# Weighted air quality aggregation
aqi = discretize_point_attributes(
    sensors,
    attributes=['pm25', 'pm10', 'ozone'],
    aggregation_funcs={
        'pm25': 'weighted_mean',
        'pm10': 'max',
        'ozone': 'mean'
    }
)
```

---

## 🔗 Utility Functions

### Create from Dictionary
```python
from DGGS.discretizer_point import create_point_feature_from_dict

data = {
    'feature_id': 'poi_001',
    'latitude': 37.7749,
    'longitude': -122.4194,
    'attributes': {'name': 'Golden Gate Park'},
    'feature_type': 'poi',
    'weight': 2.0
}

point = create_point_feature_from_dict(data)
```

### Spatial Clustering
```python
from DGGS.discretizer_point import spatial_cluster_points

# Cluster points based on distance
clusters = spatial_cluster_points(
    points,
    distance_threshold_km=0.5,  # Max distance for same cluster
    min_cluster_size=2          # Minimum points per cluster
)
```

### Calculate Statistics
```python
from DGGS.discretizer_point import calculate_point_statistics

stats = calculate_point_statistics(points, 'temperature')
# Returns: {mean, std, min, max, median, count}
```

### Filter Points
```python
from DGGS.discretizer_point import filter_points_by_attribute

# Filter by condition
hot_stations = filter_points_by_attribute(
    stations,
    'temperature',
    lambda t: t > 30  # Condition function
)
```

---

## 📏 DGGS Level Selection

| Level | Cell Size | Use Case |
|-------|-----------|----------|
| 10 | ~100 km | National/regional point networks |
| 11 | ~50 km | Regional facilities, large-scale monitoring |
| 12 | ~25 km | Typical analysis (weather stations, hospitals) |
| 13 | ~12 km | Urban analysis (POIs, infrastructure) |
| 14 | ~6 km | Fine-grained analysis (trees, sensors) |
| 15 | ~3 km | Very detailed local analysis |

**Rule of thumb**: For point data, choose level based on analysis scale and point density.

---

## 💡 Best Practices

### 1. Choose Appropriate Aggregation
```python
# Continuous variables: mean, weighted_mean
'temperature': 'weighted_mean'

# Capacity/totals: sum
'beds': 'sum'

# Quality indicators: max (best) or min (worst)
'rating': 'max'
'trauma_level': 'min'  # Lower is better

# Distribution: median (robust to outliers)
'price_level': 'median'

# Variability: std
'measurement_error': 'std'
```

### 2. Use Weights Appropriately
```python
# Quality-based weighting
station = PointFeature(..., weight=1.5)  # Higher quality data

# Population weighting
poi = PointFeature(..., weight=population_served)

# Attribute-based weighting
point = PointFeature(..., attributes={'quality': 0.8})
discretize_point_attributes(..., weight_by='quality')
```

### 3. Kernel Density for Smoothing
```python
# Simple count (no smoothing)
density = discretize_point_density(points, kernel_radius_cells=0)

# Smooth over immediate neighbors (8 neighbors)
density = discretize_point_density(points, kernel_radius_cells=1)

# More smoothing (24 neighbors)
density = discretize_point_density(points, kernel_radius_cells=2)
```

### 4. Temporal Analysis Best Practices
```python
# Use ISO format timestamps
timestamp='2024-01-15T14:30:00'

# Or custom attribute
attributes={'event_time': '2024-01-15 14:30:00'}
discretize_point_temporal(..., time_attribute='event_time')
```

---

## 📊 Comparison: Point vs Polygon vs Polyline

| Aspect | Point | Polyline | Polygon |
|--------|-------|----------|---------|
| **Geometry** | Single location | Line/path | Area |
| **Discretization** | Direct mapping | Interpolation | Coverage/centroid |
| **Aggregation** | Count/Average | Length-weighted | Area-weighted |
| **Density** | Points/km² | Length/km² | Coverage % |
| **Common Uses** | Stations, POIs, events | Roads, rivers, cables | Land parcels, watersheds |

---

## 🎨 Advanced Patterns

### Pattern 1: Multi-attribute Analysis
```python
result = discretize_point_attributes(
    points,
    attributes=['temp', 'humidity', 'pressure', 'wind_speed'],
    aggregation_funcs={
        'temp': 'weighted_mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'max'
    }
)
```

### Pattern 2: Density + Attributes
```python
# Get density
density = discretize_point_density(points, level=12)

# Get attributes
attrs = discretize_point_attributes(points, ['temperature'], level=12)

# Merge results
for token in density:
    if token in attrs:
        density[token].update(attrs[token])
```

### Pattern 3: Cluster Analysis Workflow
```python
# 1. Spatial clustering
clusters = spatial_cluster_points(points, distance_threshold_km=1.0)

# 2. Assign cluster IDs to points
for cluster in clusters:
    for point in cluster.points:
        point.cluster_id = cluster.cluster_id

# 3. Discretize with clusters
result = discretize_point_clusters(
    points,
    cluster_by='cluster_id',
    aggregate_attributes=['temperature']
)
```

### Pattern 4: Temporal Hotspot Detection
```python
# 1. Temporal analysis
temporal = discretize_point_temporal(events, time_attribute='timestamp')

# 2. Spatial clustering
hotspots = discretize_point_clusters(events, cluster_by='hotspot_id')

# 3. Density with kernel
density = discretize_point_density(
    events,
    normalize_by_area=True,
    kernel_radius_cells=1
)

# 4. Combine for spatio-temporal analysis
```

---

## 🔍 Debugging Tips

### Check Point Count
```python
points = [PointFeature(...), ...]
print(f"Total points: {len(points)}")

cells = discretize_point_features(points, level=12)
print(f"Cells covered: {len(cells)}")
print(f"Avg points/cell: {sum(c['num_points'] for c in cells.values()) / len(cells):.1f}")
```

### Verify Attributes
```python
# Check if attribute exists
for point in points:
    val = point.get_attribute('temperature')
    if val is None:
        print(f"Missing temperature for {point.feature_id}")
```

### Inspect Aggregation
```python
result = discretize_point_attributes(points, ['temp'])
for token, data in list(result.items())[:3]:
    print(f"Cell {token}:")
    print(f"  Points: {data['num_points']}")
    print(f"  Temp: {data.get('temp_mean', 'N/A')}")
```

---

## ⚡ Performance Tips

1. **Choose appropriate level**: Higher levels = more cells = slower
2. **Filter points before discretization**: Only include relevant points
3. **Limit attributes**: Only aggregate attributes you need
4. **Batch processing**: Discretize multiple point sets together when possible
5. **Kernel radius**: Keep kernel_radius_cells small (0-2) for better performance

---

## 📚 See Also

- **Related Modules**:
  - `polygon.py` - Polygon discretization
  - `polyline.py` - Polyline discretization
  - `raster.py` - Raster discretization
  - `discretized_to_kg.py` - Knowledge graph conversion

---

**Quick Reference v1.0** | December 2024
