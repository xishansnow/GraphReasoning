# DGGS spatial_utils Quick Reference

## Overview

`spatial_utils.py` is the centralized module for core spatial utilities used across all DGGS discretization modules.

## Quick Start

### Basic Imports
```python
from DGGS import (
    aggregate_values,
    discretize_aggregate,
    discretize_polygon_strict,
    calculate_distance_km,
)
```

### Statistical Aggregation
```python
# Simple aggregation
values = [1, 2, 3, 4, 5]
mean_val = aggregate_values(values, method='mean')              # 3.0
sum_val = aggregate_values(values, method='sum')               # 15.0
max_val = aggregate_values(values, method='max')               # 5.0

# Weighted aggregation
weights = [1, 2, 1, 2, 1]
weighted = aggregate_values(values, weights, 'weighted_mean')  # 3.0
```

### Discretize with Aggregation
```python
# Aggregate point observations into DGGS cells
data = [
    {'id': 'obs1', 'lat': 40.7, 'lon': -74.0, 'temp': 25.5, 'humidity': 60},
    {'id': 'obs2', 'lat': 40.71, 'lon': -74.01, 'temp': 24.8, 'humidity': 65}
]

result = discretize_aggregate(
    data,
    value_fields=['temp', 'humidity'],
    level=12,
    agg_funcs={'temp': 'mean', 'humidity': 'max'}
)

# Result: {cell_token: {'temp_mean': 25.15, 'humidity_max': 65, 'count': 2, 'entity_ids': [...]}}
```

### Weighted Discretization
```python
# Aggregate with weights (quality, confidence, importance)
data = [
    {'id': 'obs1', 'lat': 40.7, 'lon': -74.0, 'temp': 25.5, 'quality': 0.9},
    {'id': 'obs2', 'lat': 40.7, 'lon': -74.0, 'temp': 24.8, 'quality': 0.7}
]

result = discretize_weighted_aggregate(
    data,
    value_fields=['temp'],
    weight_field='quality',
    level=12
)
```

### Geometric Operations
```python
# Point-in-polygon test
polygon_ring = [(40.7, -74.0), (40.8, -74.0), (40.8, -73.9), (40.7, -73.9)]
holes = None  # or: [[(40.75, -73.95), (40.76, -73.95), (40.76, -73.94)]]

inside = point_in_polygon((40.75, -74.0), polygon_ring, holes)

# Discretize polygon to cells
polygon_geojson = [[
    [-74.0, 40.7],
    [-74.0, 40.8],
    [-73.9, 40.8],
    [-73.9, 40.7],
    [-74.0, 40.7]
]]

tokens = discretize_polygon_strict(polygon_geojson, level=13)
```

### Utility Functions
```python
# Distance calculation
dist_km = calculate_distance_km(40.7, -74.0, 40.8, -74.1)

# Bearing calculation
bearing = calculate_bearing(40.7, -74.0, 40.8, -74.1)  # 0-360 degrees

# Centroid of points
centroid = calculate_centroid([(40.7, -74.0), (40.8, -74.1)])

# Bounding box
bbox = calculate_bbox([(40.7, -74.0), (40.8, -74.1)])  # ((lat_min, lon_min), (lat_max, lon_max))
```

## Function Reference

### Statistical Aggregation

#### aggregate_values(values, weights=None, method='mean')
Aggregate a list of values using specified method.

**Methods**:
- `'mean'` - Average
- `'weighted_mean'` - Weight-weighted average (requires weights)
- `'sum'` - Sum of values
- `'max'` - Maximum value
- `'min'` - Minimum value
- `'median'` - Median value
- `'std'` - Standard deviation
- `'count'` - Count of values
- `'mode'` - Most frequent value

**Example**:
```python
aggregate_values([1, 2, 3], method='mean')  # 2.0
aggregate_values([1, 2, 3], [1, 2, 1], 'weighted_mean')  # 2.0
```

#### discretize_aggregate(data, value_fields, level=12, agg_funcs=None, entity_id_key="id")
Aggregate data values within each DGGS cell.

**Parameters**:
- `data` - Sequence of observations with 'lat', 'lon', and value fields
- `value_fields` - List of field names to aggregate (e.g., ['temp', 'humidity'])
- `level` - S2 cell level (default: 12)
- `agg_funcs` - Dict mapping field → aggregation function (str or callable)
- `entity_id_key` - Key for observation identifier

**Returns**: Dict mapping cell_token → {field: value, count: n, entity_ids: [...]}

#### discretize_weighted_aggregate(data, value_fields, weight_field, level=12, agg_method='weighted_mean', entity_id_key="id")
Aggregate data with weighted values.

**Parameters**:
- `data` - Observations with value_fields and weight_field
- `value_fields` - List of field names to aggregate
- `weight_field` - Field name containing weights
- `level` - S2 cell level (default: 12)
- `agg_method` - Aggregation method for weighted aggregation
- `entity_id_key` - Key for observation identifier

**Returns**: Dict mapping cell_token → {field: weighted_value, count: n}

### Geometric Algorithms

#### point_in_polygon(point, outer_ring, holes=None)
Test if point is inside polygon (ray casting algorithm).

**Parameters**:
- `point` - (lat, lon) tuple
- `outer_ring` - List of (lat, lon) tuples forming outer boundary
- `holes` - Optional list of hole rings

**Returns**: Boolean

#### point_in_ring(point, ring)
Test if point is inside a polygon ring.

**Parameters**:
- `point` - (lat, lon) tuple
- `ring` - List of (lat, lon) tuples forming closed ring

**Returns**: Boolean

#### segments_intersect(a1, a2, b1, b2)
Check if two line segments intersect.

**Parameters**:
- `a1, a2` - Endpoints of first segment as (lat, lon) tuples
- `b1, b2` - Endpoints of second segment as (lat, lon) tuples

**Returns**: Boolean

#### interpolate_line_segment(lat1, lon1, lat2, lon2, step_km)
Interpolate points along a line segment.

**Parameters**:
- `lat1, lon1` - Start point
- `lat2, lon2` - End point
- `step_km` - Distance between interpolated points in kilometers

**Returns**: List of (lat, lon) tuples

#### discretize_polygon_strict(polygon_coords, level=12, max_cells=1000)
Strict polygon discretization with hole support.

**Parameters**:
- `polygon_coords` - GeoJSON polygon coordinates [[lon, lat], ...]
- `level` - S2 cell level (default: 12)
- `max_cells` - Maximum candidate cells to test

**Returns**: List of cell tokens that intersect/contain the polygon

### Utility Functions

#### calculate_distance_km(lat1, lon1, lat2, lon2)
Distance between two points (haversine).

**Returns**: Distance in kilometers

#### calculate_bearing(lat1, lon1, lat2, lon2)
Bearing from point 1 to point 2.

**Returns**: Bearing in degrees (0-360, 0 is North)

#### calculate_centroid(points)
Centroid of a set of points.

**Parameters**:
- `points` - List of (lat, lon) tuples

**Returns**: (lat, lon) tuple

#### calculate_bbox(points)
Bounding box of a set of points.

**Parameters**:
- `points` - List of (lat, lon) tuples

**Returns**: ((lat_min, lon_min), (lat_max, lon_max))

## Common Patterns

### Pattern 1: Aggregate Sensor Data
```python
from DGGS import discretize_aggregate

sensor_data = [
    {'id': 'sensor_1', 'lat': 40.7, 'lon': -74.0, 'temperature': 22.5, 'humidity': 65},
    {'id': 'sensor_2', 'lat': 40.71, 'lon': -74.01, 'temperature': 23.1, 'humidity': 68},
    # ...
]

result = discretize_aggregate(
    sensor_data,
    value_fields=['temperature', 'humidity'],
    agg_funcs={
        'temperature': 'mean',
        'humidity': 'median'
    }
)
```

### Pattern 2: Quality-Weighted Aggregation
```python
from DGGS import discretize_weighted_aggregate

measurements = [
    {'id': 'meas_1', 'lat': 40.7, 'lon': -74.0, 'value': 10, 'quality': 0.95},
    {'id': 'meas_2', 'lat': 40.7, 'lon': -74.0, 'value': 11, 'quality': 0.7},
]

result = discretize_weighted_aggregate(
    measurements,
    value_fields=['value'],
    weight_field='quality'
)
```

### Pattern 3: Polygon Coverage Analysis
```python
from DGGS import discretize_polygon_strict

polygon = [[
    [-74.0, 40.7],
    [-74.0, 40.75],
    [-73.95, 40.75],
    [-73.95, 40.7],
    [-74.0, 40.7]
]]

# Get all cells covering the polygon
cell_tokens = discretize_polygon_strict(polygon, level=13)

# Each cell now represents area of polygon
print(f"Polygon covers {len(cell_tokens)} cells at level 13")
```

### Pattern 4: Distance-Based Analysis
```python
from DGGS import calculate_distance_km, calculate_bearing

# Find distance between two locations
distance = calculate_distance_km(40.7, -74.0, 51.5, -0.1)  # NYC to London
print(f"Distance: {distance:.0f} km")

# Find direction
bearing = calculate_bearing(40.7, -74.0, 51.5, -0.1)
print(f"Bearing: {bearing:.0f}°")  # 0° = North, 90° = East, etc.
```

## Integration with Generic Modules

### With point.py
```python
from DGGS.discretizer_point import PointFeature, discretize_point_features
from DGGS import aggregate_values

# Point discretization uses discretize_aggregate internally
features = [PointFeature(lat=40.7, lon=-74.0, name='POI1', attributes={'type': 'restaurant'})]
result = discretize_point_features(features, method='centroid')
```

### With polygon.py
```python
from DGGS.discretizer_polygon import PolygonFeature, discretize_polygon_features
from DGGS import discretize_polygon_strict

# Polygon discretization uses discretize_polygon_strict internally
features = [PolygonFeature(polygon_coords=[[...]], attributes={...})]
result = discretize_polygon_features(features, method='coverage')
```

### With ssurgo.py
```python
from DGGS.ssurgo import SSURGOMapUnit
from DGGS import discretize_weighted_aggregate

# SSURGO discretization uses discretize_weighted_aggregate for components
mapunit = SSURGOMapUnit(mukey='123456', ...)
result = discretize_ssurgo_features([mapunit], method='weighted')
```

## Performance Notes

- **aggregate_values**: O(n) where n = number of values
- **discretize_aggregate**: O(m + c) where m = observations, c = cells
- **discretize_polygon_strict**: O(e² ) where e = polygon edges (expensive for large polygons)
- **point_in_polygon**: O(n) where n = polygon vertices

## Backward Compatibility

All functions are also available via legacy imports:
```python
from DGGS.discretize import discretize_aggregate  # Still works
from DGGS.geometry import discretize_polygon_strict  # Still works
```

But new code should use:
```python
from DGGS import discretize_aggregate  # Recommended
from DGGS.spatial_utils import discretize_polygon_strict  # Recommended
```

## See Also

- **DGGS/point.py** - Point discretization framework
- **DGGS/polygon.py** - Polygon discretization framework
- **DGGS/polyline.py** - Polyline discretization framework
- **DGGS/raster.py** - Raster discretization framework
- **DGGS/ssurgo.py** - SSURGO soil data discretization
- **DGGS/cdl.py** - CDL crop data discretization
