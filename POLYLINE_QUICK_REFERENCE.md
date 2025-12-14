# Polyline Discretization - Quick Reference

Fast reference for discretizing line-based spatial data (roads, rivers, pipelines, trails, etc.)

---

## 🚀 Quick Start

```python
from Dggs.discretizer_polyline import (
    PolylineFeature,
    LineSegment,
    discretize_polyline_features,
    discretize_polyline_attributes
)

# 1. Create a polyline with segments
road = PolylineFeature(
    feature_id='highway_101',
    coordinates=[(40.0, -74.0), (40.1, -74.1), (40.2, -74.2)],
    segments=[
        LineSegment('seg1', (40.0, -74.0), (40.1, -74.1),
                   attributes={'speed_limit': 65, 'lanes': 4})
    ],
    feature_type='highway'
)

# 2. Discretize to DGGS cells
cells = discretize_polyline_features([road], level=12)

# 3. Aggregate attributes
attrs = discretize_polyline_attributes(
    [road],
    attributes=['speed_limit', 'lanes'],
    aggregation_funcs={'speed_limit': 'length_weighted', 'lanes': 'max'}
)
```

---

## 📊 Core Data Models

### LineSegment
```python
segment = LineSegment(
    segment_id='seg1',
    start_point=(lat1, lon1),
    end_point=(lat2, lon2),
    attributes={'attr1': value1, 'attr2': value2},
    weight=1.0  # Relative importance
)
```

### PolylineFeature
```python
feature = PolylineFeature(
    feature_id='feature_1',
    coordinates=[(lat1, lon1), (lat2, lon2), ...],
    segments=[seg1, seg2, ...],
    attributes={'name': 'Feature Name'},
    feature_type='road',  # or 'river', 'pipeline', etc.
    is_directed=True  # For one-way/flow direction
)
```

---

## 🔧 Core Functions

### 1. Basic Discretization
```python
discretize_polyline_features(
    features=[feature1, feature2],
    level=12,                    # DGGS level (10-15)
    method='interpolate',        # 'interpolate' or 'vertices'
    step_km=0.5,                # Interpolation step
    include_topology=False
)
# Returns: {cell_token: {features, feature_types, total_length_km, num_features}}
```

### 2. Attribute Aggregation
```python
discretize_polyline_attributes(
    features=[feature1, feature2],
    attributes=['attr1', 'attr2'],
    level=12,
    aggregation_funcs={
        'attr1': 'length_weighted',  # Length-weighted average
        'attr2': 'max'                # Maximum value
    }
)
# Returns: {cell_token: {attr1_length_weighted, attr2_max}}
```

### 3. Network Analysis
```python
discretize_polyline_network(
    features=[road1, road2, road3],
    level=12,
    connectivity_threshold_km=0.1
)
# Returns: {cell_token: {is_node, node_type, connected_features, num_connections}}
```

### 4. Density Calculation
```python
discretize_polyline_density(
    features=[roads],
    level=12,
    normalize_by_area=True  # Return km/km² instead of km
)
# Returns: {cell_token: {total_length_km, density_km_per_km2, feature_count}}
```

### 5. Flow Analysis
```python
discretize_polyline_flow(
    features=[rivers],
    flow_attribute='discharge_m3s',
    level=12,
    accumulate=True
)
# Returns: {cell_token: {total_flow, avg_flow, max_flow, min_flow}}
```

---

## 📐 Aggregation Methods

Use with `get_weighted_attribute()` or `discretize_polyline_attributes()`:

| Method | Description | Use Case |
|--------|-------------|----------|
| `weighted_mean` | Weight-weighted average | General weighted average |
| `length_weighted` | Length-weighted average | Speed limits, gradients |
| `mean` | Simple average | Equal importance |
| `sum` | Sum of values | Traffic volume, flow |
| `max` | Maximum value | Max speed, peak flow |
| `min` | Minimum value | Min clearance |
| `median` | Median value | Robust to outliers |

---

## 🎯 Common Use Cases

### Roads & Transportation
```python
road = PolylineFeature(
    'highway_1',
    coordinates,
    segments=[
        LineSegment('seg1', start, end,
                   attributes={
                       'speed_limit': 65,
                       'lanes': 4,
                       'traffic_volume': 5000,
                       'surface_type': 'asphalt'
                   })
    ],
    feature_type='highway',
    is_directed=True
)

# Aggregate road attributes
road_attrs = discretize_polyline_attributes(
    [road],
    attributes=['speed_limit', 'lanes', 'traffic_volume'],
    aggregation_funcs={
        'speed_limit': 'length_weighted',
        'lanes': 'max',
        'traffic_volume': 'sum'
    }
)
```

### Rivers & Hydrology
```python
river = PolylineFeature(
    'mississippi',
    coordinates,
    segments=[
        LineSegment('reach1', start, end,
                   attributes={
                       'discharge_m3s': 150,
                       'width_m': 50,
                       'depth_m': 3.5
                   })
    ],
    feature_type='river',
    is_directed=True
)

# Analyze flow
flow = discretize_polyline_flow(river, flow_attribute='discharge_m3s')

# Calculate sinuosity
from Dggs.discretizer_polyline import calculate_polyline_sinuosity
sinuosity = calculate_polyline_sinuosity(river)
# 1.0 = straight, >1.3 = meandering
```

### Pipelines & Utilities
```python
pipeline = PolylineFeature(
    'gas_main_1',
    coordinates,
    segments=[
        LineSegment('pipe_seg1', start, end,
                   attributes={
                       'diameter_inches': 36,
                       'pressure_psi': 1000,
                       'flow_rate_mcf': 500,
                       'material': 'steel'
                   })
    ],
    feature_type='gas_pipeline',
    is_directed=True
)

# Calculate capacity
capacity = discretize_polyline_attributes(
    [pipeline],
    attributes=['diameter_inches', 'flow_rate_mcf'],
    aggregation_funcs={
        'diameter_inches': 'length_weighted',
        'flow_rate_mcf': 'mean'
    }
)
```

### Trails & Recreation
```python
trail = PolylineFeature(
    'appalachian_section_5',
    coordinates,
    segments=[
        LineSegment('section1', start, end,
                   attributes={
                       'elevation_m': 650,
                       'difficulty': 'difficult',
                       'grade_pct': 15
                   })
    ],
    feature_type='hiking_trail'
)

# Fine-resolution discretization
cells = discretize_polyline_features([trail], level=14, step_km=0.1)

# Elevation profile
elevation = discretize_polyline_attributes(
    [trail],
    attributes=['elevation_m', 'grade_pct'],
    aggregation_funcs={'elevation_m': 'mean', 'grade_pct': 'max'}
)
```

---

## 🔗 Utility Functions

### Create from Dictionary
```python
from Dggs.discretizer_polyline import create_polyline_feature_from_dict

data = {
    'feature_id': 'highway_95',
    'coordinates': [(lat1, lon1), (lat2, lon2)],
    'segments': [
        {
            'segment_id': 'seg1',
            'start_point': (lat1, lon1),
            'end_point': (lat2, lon2),
            'attributes': {'speed_limit': 70},
            'weight': 1.0
        }
    ],
    'feature_type': 'interstate',
    'is_directed': True
}

feature = create_polyline_feature_from_dict(data)
```

### Calculate Sinuosity
```python
from Dggs.discretizer_polyline import calculate_polyline_sinuosity

sinuosity = calculate_polyline_sinuosity(feature)
# 1.0 = perfectly straight
# 1.0-1.2 = relatively straight
# 1.2-1.5 = winding
# >1.5 = highly meandering
```

### Merge Networks
```python
from Dggs.discretizer_polyline import merge_polyline_networks

merged = merge_polyline_networks(
    [network1, network2],
    merge_strategy='union'  # or 'intersection'
)
```

---

## 🎨 Method Comparison

### Interpolate vs Vertices

**Interpolate** (default):
```python
cells = discretize_polyline_features(
    [road],
    method='interpolate',
    step_km=0.5  # Points every 0.5 km
)
```
- ✅ More accurate for curved lines
- ✅ Better coverage
- ❌ Slower

**Vertices**:
```python
cells = discretize_polyline_features(
    [road],
    method='vertices'  # Use only line vertices
)
```
- ✅ Faster (2-3x)
- ✅ Good for simple straight lines
- ❌ May miss cells for curved lines

---

## 📏 DGGS Level Selection

Choose level based on analysis scale:

| Level | Cell Size | Use Case |
|-------|-----------|----------|
| 10 | ~100 km | Continental/regional analysis |
| 11 | ~50 km | Regional networks |
| 12 | ~25 km | Typical analysis (default) |
| 13 | ~12 km | Local/urban analysis |
| 14 | ~6 km | Detailed local analysis |
| 15 | ~3 km | Fine-grained analysis |

**Rule of thumb**: Choose level where cell size is 10-50% of average feature length.

---

## 🚦 Network Topology

### Node Types

**Junction**: Intersection of 3+ features
```python
network = discretize_polyline_network([road1, road2, road3])
junctions = {k: v for k, v in network.items() if v['node_type'] == 'junction'}
```

**Endpoint**: Start or end of a feature
```python
endpoints = {k: v for k, v in network.items() if v['node_type'] == 'endpoint'}
```

**Through**: Regular segment (no intersection)
```python
through = {k: v for k, v in network.items() if v['node_type'] == 'through'}
```

---

## 💡 Best Practices

### 1. Choose Appropriate Step Size
```python
# For highways (fast, straight)
step_km=1.0

# For urban roads (more detail needed)
step_km=0.5

# For trails (fine detail)
step_km=0.1
```

### 2. Use Length-Weighted for Continuous Attributes
```python
# Speed limit varies along road
'speed_limit': 'length_weighted'

# Elevation changes along trail
'elevation_m': 'length_weighted'
```

### 3. Use Sum for Accumulative Attributes
```python
# Total traffic volume
'traffic_volume': 'sum'

# Total flow
'discharge_m3s': 'sum'
```

### 4. Use Max for Capacity Constraints
```python
# Maximum lanes needed
'lanes': 'max'

# Peak flow
'peak_flow': 'max'
```

### 5. Set Directed Flag Appropriately
```python
# One-way roads
is_directed=True

# Rivers (flow direction matters)
is_directed=True

# Hiking trails (bidirectional)
is_directed=False
```

---

## 📦 Integration Examples

### With Knowledge Graph
```python
from Dggs.discretizer_polyline import discretize_polyline_features

# Discretize roads
cells = discretize_polyline_features([road1, road2], level=12)

# Convert to triplets
triplets = []
for cell_token, data in cells.items():
    for feature_id in data['features']:
        triplets.append((feature_id, 'intersects_cell', cell_token))
        triplets.append((cell_token, 'has_feature_type', data['feature_types'][0]))
```

### With Pandas
```python
import pandas as pd

# Discretize with attributes
result = discretize_polyline_attributes([road], attributes=['speed_limit'])

# Convert to DataFrame
df = pd.DataFrame([
    {'cell': token, 'speed_limit': data.get('speed_limit_length_weighted')}
    for token, data in result.items()
])
```

### Export to GeoJSON
```python
import json

# Discretize
cells = discretize_polyline_features([road], level=12)

# Convert to GeoJSON (simplified)
features = []
for token, data in cells.items():
    # Get cell center (would need lat/lon conversion)
    features.append({
        'type': 'Feature',
        'properties': data,
        'geometry': {
            'type': 'Point',
            'coordinates': [lon, lat]  # From cell token
        }
    })

geojson = {'type': 'FeatureCollection', 'features': features}
```

---

## 🔍 Debugging Tips

### Check Feature Length
```python
feature = PolylineFeature(...)
print(f"Total length: {feature.total_length_km:.2f} km")
print(f"Segments: {len(feature.segments)}")
```

### Verify Interpolation
```python
points = feature.interpolate_points(step_km=0.5)
print(f"Interpolated to {len(points)} points")
```

### Inspect Cell Coverage
```python
cells = discretize_polyline_features([feature], level=12)
print(f"Covers {len(cells)} cells")
for token, data in list(cells.items())[:5]:
    print(f"  {token}: {data}")
```

### Check Attribute Aggregation
```python
# Get weighted attribute directly from feature
value = feature.get_weighted_attribute('speed_limit', 'length_weighted')
print(f"Weighted speed limit: {value}")
```

---

## ⚡ Performance Tips

1. **Use vertices method for simple lines**: 2-3x faster
2. **Increase step_km for coarse analysis**: Fewer points = faster
3. **Lower DGGS level for regional analysis**: Fewer cells = faster
4. **Filter attributes before discretization**: Only compute what you need
5. **Batch process multiple features**: Single discretization call is more efficient

---

## 🎓 Common Patterns

### Pattern 1: Multi-attribute Analysis
```python
# Discretize once, get multiple attributes
result = discretize_polyline_attributes(
    roads,
    attributes=['speed_limit', 'lanes', 'traffic_volume', 'surface_quality'],
    aggregation_funcs={
        'speed_limit': 'length_weighted',
        'lanes': 'max',
        'traffic_volume': 'sum',
        'surface_quality': 'mean'
    }
)
```

### Pattern 2: Network + Attributes
```python
# Get topology
network = discretize_polyline_network(roads, level=12)

# Get attributes
attrs = discretize_polyline_attributes(roads, ['traffic_volume'], level=12)

# Merge results
for token in network:
    if token in attrs:
        network[token].update(attrs[token])
```

### Pattern 3: Multi-scale Analysis
```python
# Regional overview
regional = discretize_polyline_features(rivers, level=10)

# Local detail
local = discretize_polyline_features(rivers, level=14)

# Compare density at different scales
```

---

## 📚 See Also

- **Full Documentation**: `POLYLINE_REFACTORING_SUMMARY.md`
- **Examples**: `examples/polyline_examples.py`
- **Related Modules**:
  - `polygon.py` - Polygon discretization
  - `raster.py` - Raster discretization
  - `geometry.py` - Basic geometric operations
  - `discretized_to_kg.py` - Knowledge graph conversion

---

**Quick Reference v1.0** | December 2024
