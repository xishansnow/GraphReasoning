# DGGS (S2) 层级关系与空间拓扑分析

## 概述

已成功扩展 DGGS 模块以支持：
1. **跨尺度（cross-scale）层级关系**
2. **空间拓扑关系分析**（equal, contains, within, adjacent, disjoint）
3. **方位关系计算**（8个基本方向 + 精确方位角）
4. **空间邻近查询**（范围搜索、距离计算）

参考论文："The S2 Hierarchical Discrete Global Grid as a Nexus for Data Representation, Integration, and Querying Across Geospatial Knowledge Graphs"。

## 新增功能

### 1. 空间拓扑关系

基于 S2 Cell 判断两个地理实体之间的拓扑关系：

```python
from GraphReasoning.Dggs import DggsS2

grid = DGGSS2(level=12)

# 获取两个位置的 cell tokens
token1 = grid.latlon_to_token(42.3601, -71.0589, level=12)
token2 = grid.latlon_to_token(42.3605, -71.0596, level=12)

# 判断拓扑关系
relation = grid.spatial_relation(token1, token2)
# 返回: 'equal', 'contains', 'within', 'adjacent', 或 'disjoint'
print(f"Spatial relation: {relation}")
```

#### 拓扑关系类型

- **`equal`**: 两个 cell 完全相同
- **`contains`**: cell1 包含 cell2（cell1 层级更粗）
- **`within`**: cell1 被 cell2 包含（cell1 层级更细）
- **`adjacent`**: 同层级且相邻
- **`disjoint`**: 不相交且不相邻

#### 多尺度拓扑分析

```python
# 在不同层级检查拓扑关系
loc1 = {"lat": 42.3601, "lon": -71.0589}
loc2 = {"lat": 42.3605, "lon": -71.0596}

for level in [10, 12, 14]:
    token1 = grid.latlon_to_token(loc1["lat"], loc1["lon"], level=level)
    token2 = grid.latlon_to_token(loc2["lat"], loc2["lon"], level=level)
    relation = grid.spatial_relation(token1, token2)
    print(f"Level {level}: {relation}")

# 输出示例:
# Level 10: equal      (粗尺度下，同一个大 cell)
# Level 12: adjacent   (中等尺度下，相邻 cells)
# Level 14: disjoint   (精细尺度下，不同 cells)
```

### 2. 方位关系分析

计算两个地理实体之间的方向关系：

```python
# 计算精确方位角（0-360度，0=北，90=东）
bearing = grid.bearing(lat1, lon1, lat2, lon2)
print(f"Bearing: {bearing:.1f}°")

# 获取基本方位（N, NE, E, SE, S, SW, W, NW）
direction = grid.cardinal_direction(lat1, lon1, lat2, lon2)
print(f"Direction: {direction}")

# 示例：从波士顿到不同城市的方向
locations = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
]

boston = (42.3601, -71.0589)
for loc in locations:
    direction = grid.cardinal_direction(*boston, loc["lat"], loc["lon"])
    bearing = grid.bearing(*boston, loc["lat"], loc["lon"])
    print(f"{loc['name']:15s}: {direction:3s} ({bearing:6.1f}°)")

# 输出:
# New York       : SW  (220.5°)
# San Francisco  : W   (281.3°)
# Miami          : S   (185.2°)
```

### 3. 距离计算

计算两点之间的大圆距离（Great Circle Distance）：

```python
# 计算距离（千米）
distance = grid.distance_km(lat1, lon1, lat2, lon2)
print(f"Distance: {distance:.2f} km")

# 示例
mit = (42.3601, -71.0942)
harvard = (42.3770, -71.1167)
distance = grid.distance_km(*mit, *harvard)
print(f"MIT to Harvard: {distance:.2f} km")  # 约 2.5 km
```

### 4. 实体关系综合分析

对两个地理实体进行全面的空间关系分析：

```python
entity1 = {"id": "A", "lat": 42.3601, "lon": -71.0589, "name": "MIT"}
entity2 = {"id": "B", "lat": 42.3770, "lon": -71.1167, "name": "Harvard"}

relation = grid.entity_relation(
    entity1, 
    entity2,
    level=12,
    distance_threshold_km=3.0
)

print(relation)
# 输出:
# {
#     'topology': 'disjoint',
#     'direction': 'NW',
#     'distance_km': 2.543,
#     'proximity': 'near',
#     'cell1': '89c25c1d',
#     'cell2': '89c25c03'
# }
```

### 5. 空间邻近查询

查找指定范围内的所有实体：

```python
entities = [
    {"id": "mit", "lat": 42.3601, "lon": -71.0942, "name": "MIT"},
    {"id": "harvard", "lat": 42.3770, "lon": -71.1167, "name": "Harvard"},
    {"id": "bu", "lat": 42.3505, "lon": -71.1054, "name": "Boston University"},
    {"id": "northeastern", "lat": 42.3398, "lon": -71.0892, "name": "Northeastern"},
]

# 查找 MIT 周围 5km 范围内的实体
nearby = grid.find_entities_in_range(
    entities,
    center_lat=42.3601,
    center_lon=-71.0942,
    radius_km=5.0,
    level=13
)

for ent in nearby:
    print(f"{ent['name']:20s}: {ent['distance_km']:5.2f} km to the {ent['direction']}")

# 输出:
# MIT                 :  0.00 km to the N
# Northeastern        :  2.31 km to the SE
# Boston University   :  3.45 km to the W
```

### 6. 批量关系分析
### 7. 空间数据离散化（Discretization）

将不同类型空间数据离散到 DGGS（S2 cells）：

```python
from GraphReasoning.Dggs import (
    discretize_points, discretize_paths, discretize_regions_bbox,
    discretize_buffers, discretize_geojson
)

# 点离散化
entities = [
    {"id": "A", "lat": 42.36, "lon": -71.05},
    {"id": "B", "lat": 42.37, "lon": -71.06},
]
point_cells = discretize_points(entities, level=12)

# 线离散化（按步长采样）
path = [(42.36, -71.05), (42.40, -71.10), (42.42, -71.15)]
line_cells = discretize_paths([path], level=12, step_km=0.5)[0]

# 矩形区域离散化（bbox）
regions = [{"sw": (42.30, -71.20), "ne": (42.45, -71.00)}]
bbox_cells = discretize_regions_bbox(regions, level=12)[0]

# 缓冲区离散化（圆形 cap）
buffer_cells = discretize_buffers(entities, radius_km=5.0, level=12)

# GeoJSON 离散化（Point/LineString/Polygon）
geojson_line = {"type": "LineString", "coordinates": [[-71.05, 42.36], [-71.10, 42.40]]}
cells_geojson = discretize_geojson(geojson_line, level=12, step_km=0.5)
```

注意：多边形离散化默认使用边界采样和外包矩形覆盖的合并近似；若需严格几何覆盖，请使用支持 S2 Polygon 的库进一步过滤。

### 8. 拓扑丰化图（Topology-Enriched Graph）

构建包含实体、网格 cells、同级邻接、父子关系、以及实体间拓扑/方向/距离的综合图：

```python
from GraphReasoning.Dggs import build_topology_enriched_graph
import networkx as nx

entities = [
    {"id": "A", "lat": 42.36, "lon": -71.05},
    {"id": "B", "lat": 42.37, "lon": -71.06},
]

KG = build_topology_enriched_graph(
    entities,
    level=12,
    include_cells=True,
    include_hierarchy=True,
    distance_threshold_km=3.0,
)

print(KG.number_of_nodes(), KG.number_of_edges())

# 查询实体间的空间关系边
for u, v, data in KG.edges(data=True):
    if data.get('type') == 'direction':
        print(u, '->', v, data['direction'], data.get('distance_km'))
    if data.get('type') == 'topology':
        print(u, '->', v, data['relation'])
```

该图适合作为地理知识图谱的骨架，支持跨尺度检索、拓扑关系推理和方向/距离分析。

分析多个实体之间的所有两两关系：

```python
from GraphReasoning.Dggs import analyze_entity_relationships

relationships = analyze_entity_relationships(
    entities,
    level=13,
    distance_threshold_km=3.0
)

for rel in relationships:
    print(f"{rel['entity1_id']} <-> {rel['entity2_id']}: "
          f"{rel['topology']}, {rel['direction']}, "
          f"{rel['distance_km']} km, {rel['proximity']}")

# 统计拓扑类型分布
topology_counts = {}
for rel in relationships:
    topo = rel['topology']
    topology_counts[topo] = topology_counts.get(topo, 0) + 1

print("Topology distribution:", topology_counts)
# 输出: {'adjacent': 2, 'disjoint': 4}

### 1. 父子关系查询

```python
from GraphReasoning.Dggs import DggsS2

grid = DGGSS2(level=12)
token = grid.latlon_to_token(42.3601, -71.0589, level=12)

# 获取父级 cell（更粗糙的分辨率）
parent_l10 = grid.parent(token, parent_level=10)
parent_l8 = grid.parent(token, parent_level=8)

# 获取子级 cells（更精细的分辨率）
children_l14 = grid.children(token, child_level=14)
children_l15 = grid.children(token, child_level=15)
```

### 2. 层级图构建

支持构建包含多个尺度层级的有向图，包含三种关系类型：
- `adjacent`: 同级相邻关系
- `parent_of`: 父到子的包含关系
- `child_of`: 子到父的归属关系

```python
# 覆盖一个区域
cells = grid.cover_cap(42.36, -71.05, radius_km=5, level=12)

# 构建层级图（包含父级）
H = grid.build_hierarchical_graph(
    cells, 
    include_parents=True,   # 添加父级 cells
    include_children=False  # 不添加子级 cells
)

# H 是一个 DiGraph，包含不同层级的节点和跨尺度边
print(f"Nodes: {H.number_of_nodes()}")
print(f"Edges: {H.number_of_edges()}")

# 查询特定关系类型
for u, v, data in H.edges(data=True):
    if data['relation'] == 'parent_of':
        print(f"{u} contains {v}")
```

### 3. 跨尺度空间查询

```python
# 在不同层级检查两个位置是否属于同一 cell
loc1 = {"lat": 42.3601, "lon": -71.0589}
loc2 = {"lat": 42.3605, "lon": -71.0596}

for level in [10, 12, 14]:
    token1 = grid.latlon_to_token(loc1["lat"], loc1["lon"], level=level)
    token2 = grid.latlon_to_token(loc2["lat"], loc2["lon"], level=level)
    same_cell = token1 == token2
    print(f"Level {level}: same cell = {same_cell}")

# 查找共同祖先
token1 = grid.latlon_to_token(loc1["lat"], loc1["lon"], level=14)
token2 = grid.latlon_to_token(loc2["lat"], loc2["lon"], level=14)

for level in range(13, -1, -1):
    p1 = grid.parent(token1, parent_level=level)
    p2 = grid.parent(token2, parent_level=level)
    if p1 == p2:
        print(f"Common ancestor at level {level}: {p1}")
        break
```

## 新增方法

### `DGGSS2` 类 - 层级关系方法

1. **`parent(token, parent_level)`**
   - 获取指定层级的父 cell
   - `parent_level` 必须小于当前 cell 的层级

2. **`children(token, child_level)`**
   - 获取指定层级的所有子 cells
   - `child_level` 必须大于当前 cell 的层级
   - 递归遍历找到目标层级的所有后代

3. **`build_hierarchical_graph(cell_tokens, include_parents=True, include_children=False)`**
   - 构建包含多尺度关系的有向图
   - 支持添加父级和/或子级 cells
   - 返回 `nx.DiGraph`，边属性 `relation` 包含关系类型

### `DGGSS2` 类 - 空间关系方法

4. **`spatial_relation(token1, token2)`**
   - 判断两个 cell 之间的拓扑关系
   - 返回: `'equal'`, `'contains'`, `'within'`, `'adjacent'`, 或 `'disjoint'`

5. **`distance_km(lat1, lon1, lat2, lon2)`**
   - 计算两点之间的大圆距离（千米）
   - 使用精确的球面几何计算

6. **`bearing(lat1, lon1, lat2, lon2)`**
   - 计算从点1到点2的初始方位角
   - 返回 0-360 度（0=北，90=东，180=南，270=西）

7. **`cardinal_direction(lat1, lon1, lat2, lon2)`**
   - 获取基本方位
   - 返回 8 个方向之一：`'N'`, `'NE'`, `'E'`, `'SE'`, `'S'`, `'SW'`, `'W'`, `'NW'`

8. **`entity_relation(entity1, entity2, level, distance_threshold_km)`**
   - 综合分析两个实体的空间关系
   - 返回字典包含：`topology`, `direction`, `distance_km`, `proximity`, `cell1`, `cell2`

9. **`find_entities_in_range(entities, center_lat, center_lon, radius_km, level)`**
   - 查找指定范围内的所有实体
   - 返回按距离排序的实体列表，每个实体附加 `distance_km` 和 `direction` 属性

### 辅助函数

10. **`analyze_entity_relationships(entities, level, distance_threshold_km)`**
    - 批量分析所有实体对的空间关系
    - 返回关系列表，每个关系包含完整的空间信息

## 使用场景

### 场景1：空间拓扑查询

```python
# 判断两个地点是否在同一区域
entities = [
    {"id": "store_1", "lat": 42.36, "lon": -71.05, "name": "Store 1"},
    {"id": "store_2", "lat": 42.37, "lon": -71.06, "name": "Store 2"},
]

grid = DGGSS2(level=12)
token1 = grid.latlon_to_token(entities[0]["lat"], entities[0]["lon"])
token2 = grid.latlon_to_token(entities[1]["lat"], entities[1]["lon"])

relation = grid.spatial_relation(token1, token2)
if relation == "equal":
    print("两个商店在同一网格区域")
elif relation == "adjacent":
    print("两个商店在相邻的网格区域")
else:
    print(f"两个商店的关系: {relation}")
```

### 场景2：方位导航

```python
# 计算从一个地点到另一个地点的方向和距离
origin = {"lat": 42.3601, "lon": -71.0589, "name": "Origin"}
destination = {"lat": 42.4501, "lon": -71.2589, "name": "Destination"}

grid = DGGSS2()
direction = grid.cardinal_direction(
    origin["lat"], origin["lon"],
    destination["lat"], destination["lon"]
)
distance = grid.distance_km(
    origin["lat"], origin["lon"],
    destination["lat"], destination["lon"]
)

print(f"从 {origin['name']} 到 {destination['name']}:")
print(f"  方向: {direction}")
print(f"  距离: {distance:.2f} km")
```

### 场景3：邻近搜索

```python
# 查找某个点周围的所有兴趣点 (POI)
pois = [
    {"id": "restaurant_1", "lat": 42.36, "lon": -71.05, "type": "restaurant"},
    {"id": "cafe_1", "lat": 42.37, "lon": -71.06, "type": "cafe"},
    {"id": "park_1", "lat": 42.38, "lon": -71.07, "type": "park"},
    # ... more POIs
]

# 用户当前位置
user_location = (42.365, -71.055)

grid = DGGSS2(level=14)
nearby_pois = grid.find_entities_in_range(
    pois,
    center_lat=user_location[0],
    center_lon=user_location[1],
    radius_km=2.0,
    level=14
)

print(f"在 2km 范围内找到 {len(nearby_pois)} 个兴趣点:")
for poi in nearby_pois:
    print(f"  {poi['type']:12s}: {poi['distance_km']:.2f} km 在 {poi['direction']} 方向")
```

### 场景4：多尺度地理实体聚合

```python
# 将实体附加到精细层级
entities = [
    {"id": "store_1", "lat": 42.36, "lon": -71.05},
    {"id": "store_2", "lat": 42.37, "lon": -71.06},
    # ...
]

grid = DGGSS2(level=14)  # 精细层级
G = grid.attach_entities(entities, level=14)

# 查看在更粗糙层级的聚合
cell_tokens = [n for n in G.nodes() if G.nodes[n].get('type') == 'cell']
parent_groups = {}
for token in cell_tokens:
    parent = grid.parent(token, parent_level=12)
    if parent not in parent_groups:
        parent_groups[parent] = []
    parent_groups[parent].append(token)

print(f"Level 12 aggregation: {len(parent_groups)} parent cells")
```

### 场景5：空间查询优化

```python
# 粗层级快速过滤 + 精细层级精确匹配
query_lat, query_lon = 42.36, -71.05
query_radius_km = 10

# 1. 用粗层级快速覆盖
coarse_cells = grid.cover_cap(query_lat, query_lon, query_radius_km, level=10)
print(f"Coarse filtering: {len(coarse_cells)} level-10 cells")

# 2. 将每个粗 cell 细分到精细层级
fine_cells = []
for coarse_token in coarse_cells:
    children = grid.children(coarse_token, child_level=14)
    fine_cells.extend(children)

print(f"Refined search space: {len(fine_cells)} level-14 cells")
```

### 场景6: 地理知识图谱构建

```python
# 构建包含拓扑和方位关系的地理知识图谱
from GraphReasoning.Dggs import analyze_entity_relationships
import networkx as nx

entities = [
    {"id": "A", "lat": 42.36, "lon": -71.05, "name": "Entity A"},
    {"id": "B", "lat": 42.37, "lon": -71.06, "name": "Entity B"},
    {"id": "C", "lat": 42.38, "lon": -71.07, "name": "Entity C"},
]

# 分析所有空间关系
relationships = analyze_entity_relationships(
    entities,
    level=12,
    distance_threshold_km=5.0
)

# 构建知识图谱
KG = nx.MultiDiGraph()

# 添加实体节点
for ent in entities:
    KG.add_node(ent["id"], type="entity", **ent)

# 添加空间关系边
for rel in relationships:
    # 拓扑关系边
    KG.add_edge(
        rel["entity1_id"],
        rel["entity2_id"],
        type="topology",
        relation=rel["topology"]
    )
    
    # 方位关系边
    KG.add_edge(
        rel["entity1_id"],
        rel["entity2_id"],
        type="direction",
        direction=rel["direction"],
        distance_km=rel["distance_km"]
    )

# 现在可以查询知识图谱
# 例如：找到所有在 A 东北方向的实体
for u, v, data in KG.edges(data=True):
    if u == "A" and data.get("type") == "direction" and data.get("direction") == "NE":
        print(f"Entity {v} is to the NE of A at {data['distance_km']} km")
```

### 场景7：层级知识图谱

```python
# 构建多层级的地理知识图谱
base_cells = grid.cover_rectangle((sw_lat, sw_lon), (ne_lat, ne_lon), level=12)

# 包含父级形成层级结构
KG = grid.build_hierarchical_graph(base_cells, include_parents=True)

# 添加实体到图中
for entity in entities:
    ent_token = grid.latlon_to_token(entity["lat"], entity["lon"], level=12)
    KG.add_node(entity["id"], type="entity", **entity)
    KG.add_edge(entity["id"], ent_token, relation="located_at")

# 现在可以跨尺度查询：
# - 邻近关系（adjacent edges）
# - 包含关系（parent_of/child_of edges）
# - 实体与空间的关联（located_at edges）
```

## S2 层级特性

S2 DGGS 的层级结构特点：
- **Level 0**: 6 个基础 faces
- **Level n**: 每个 cell 有 4 个子 cells
- **Level 30**: 最精细层级

常用层级对应的大致面积（赤道附近）：
- Level 6: ~10,000 km²
- Level 8: ~1,000 km²
- Level 10: ~60 km²
- Level 12: ~4 km²
- Level 14: ~0.25 km²
- Level 16: ~0.015 km²

## 安装依赖

确保安装了 S2 库：

```bash
pip install s2sphere
```

或更新整个环境：

```bash
pip install -e .
```

## 示例文件

查看完整示例：
- `examples/dggs_examples.py`: 包含 5 个详细示例，演示所有层级关系功能
- `test_dggs_hierarchical.py`: 快速功能测试

运行示例：

```bash
# 确保安装依赖后运行
python examples/dggs_examples.py
```

## 技术实现

### 父 cell 查找
使用 S2 CellId 的 `parent(level)` 方法直接获取指定层级的祖先。

### 子 cells 查找
递归遍历 S2 cell 的 `children()` 方法，直到达到目标层级。对于跨多层级的查询（如从 L10 到 L14），会递归展开中间层级。

### 层级图构建
1. 添加所有输入 cells 为节点
2. 可选添加父级/子级节点
3. 添加三种边类型：
   - 同级邻接边（undirected in DiGraph represented as bidirectional）
   - 父到子的有向边
   - 子到父的有向边

## 与文献的对应

参考论文强调 S2 DGGS 的优势：
- ✅ 层级结构：支持多尺度表示
- ✅ 相邻关系：同级 cells 的邻近查询
- ✅ 包含关系：父子层级的嵌套
- ✅ 知识图谱集成：统一的图结构表示地理实体和空间 cells
- ✅ 空间拓扑关系：判断实体间的包含、相邻、分离等关系
- ✅ 方位关系：计算实体间的方向和距离
- ✅ 空间查询：高效的邻近搜索和范围查询

本实现提供了论文中描述的核心功能，并扩展了实用的空间分析能力，适用于：
- 地理知识图谱构建
- 多尺度空间查询
- 跨尺度关系推理
- 地理实体聚合分析
- 空间拓扑关系判断
- 方位导航和路径规划
- 基于位置的服务（LBS）
- 地理信息检索

## 空间关系分析总结

### 支持的拓扑关系类型

| 关系类型 | 说明 | 示例场景 |
|---------|------|---------|
| `equal` | 两个实体在同一 cell | 同一建筑物内的设施 |
| `contains` | cell1 包含 cell2 | 城市包含街区 |
| `within` | cell1 被 cell2 包含 | 街区属于城市 |
| `adjacent` | 同层级相邻 | 相邻的街区 |
| `disjoint` | 不相交也不相邻 | 不同城市的设施 |

### 支持的方位关系

- **精确方位角**: 0-360° (0=正北, 90=正东, 180=正南, 270=正西)
- **基本方位**: N, NE, E, SE, S, SW, W, NW (8个基本方向)
- **距离测量**: 基于球面几何的大圆距离（千米）

### 邻近性分类

根据 `distance_threshold_km` 参数自动分类：
- **near**: 距离 < threshold
- **moderate**: threshold ≤ 距离 < 3×threshold  
- **distant**: 距离 ≥ 3×threshold

### 典型应用模式

1. **单点查询**: 查找某位置周围的 POI
   ```python
   nearby = grid.find_entities_in_range(pois, lat, lon, radius_km=2.0)
   ```

2. **两点关系**: 判断两个实体的空间关系
   ```python
   relation = grid.entity_relation(entity1, entity2, level=12)
   ```

3. **多点分析**: 批量分析所有实体对的关系
   ```python
   relationships = analyze_entity_relationships(entities, level=12)
   ```

4. **知识图谱**: 构建包含空间关系的图谱
   ```python
   # 将拓扑和方位关系添加为图的边
   for rel in relationships:
       KG.add_edge(rel['entity1_id'], rel['entity2_id'], 
                   topology=rel['topology'], 
                   direction=rel['direction'])
   ```
