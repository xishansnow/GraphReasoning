# DGGS 空间关系分析快速指南

## 核心功能速查

### 1. 拓扑关系判断

```python
from GraphReasoning.dggs import DGGSS2

grid = DGGSS2(level=12)

# 获取两个位置的 cell
token1 = grid.latlon_to_token(lat1, lon1)
token2 = grid.latlon_to_token(lat2, lon2)

# 判断拓扑关系
relation = grid.spatial_relation(token1, token2)
# 返回: 'equal', 'contains', 'within', 'adjacent', 'disjoint'
```

### 2. 方位和距离

```python
# 距离（千米）
distance = grid.distance_km(lat1, lon1, lat2, lon2)

# 方位角（0-360度）
bearing = grid.bearing(lat1, lon1, lat2, lon2)

# 基本方向（N/NE/E/SE/S/SW/W/NW）
direction = grid.cardinal_direction(lat1, lon1, lat2, lon2)
```

### 3. 实体关系综合分析

```python
entity1 = {"id": "A", "lat": 42.36, "lon": -71.05}
entity2 = {"id": "B", "lat": 42.37, "lon": -71.06}

relation = grid.entity_relation(entity1, entity2, level=12, distance_threshold_km=3.0)

# 返回:
# {
#     'topology': 'adjacent',      # 拓扑关系
#     'direction': 'NE',          # 方向
#     'distance_km': 1.234,       # 距离
#     'proximity': 'near',        # 邻近度
#     'cell1': '89c25c1d',       # entity1 的 cell
#     'cell2': '89c25c03'        # entity2 的 cell
# }
```

### 4. 邻近搜索

```python
# 查找 5km 范围内的所有实体
nearby = grid.find_entities_in_range(
    entities,
    center_lat=42.36,
    center_lon=-71.05,
    radius_km=5.0,
    level=13
)

# 返回按距离排序的实体列表，每个包含:
# - 原有的实体属性
# - distance_km: 到中心点的距离
# - direction: 相对于中心点的方向
```

### 5. 批量关系分析

```python
from GraphReasoning.dggs import analyze_entity_relationships

relationships = analyze_entity_relationships(
    entities,
    level=12,
    distance_threshold_km=3.0
)

# 返回所有实体对的关系列表
# 每个关系包含: entity1_id, entity2_id, topology, direction, distance_km, proximity, cell1, cell2
```

## 典型应用示例

### 示例 1: 判断两个商店是否在同一区域

```python
grid = DGGSS2(level=12)

store_a = {"lat": 42.36, "lon": -71.05}
store_b = {"lat": 42.37, "lon": -71.06}

token_a = grid.latlon_to_token(store_a["lat"], store_a["lon"])
token_b = grid.latlon_to_token(store_b["lat"], store_b["lon"])

relation = grid.spatial_relation(token_a, token_b)

if relation == "equal":
    print("两个商店在同一网格区域")
elif relation == "adjacent":
    print("两个商店在相邻区域")
else:
    print(f"关系: {relation}")
```

### 示例 2: 导航指引

```python
grid = DGGSS2()

origin = (42.3601, -71.0589)  # 起点
destination = (42.4501, -71.2589)  # 终点

direction = grid.cardinal_direction(*origin, *destination)
distance = grid.distance_km(*origin, *destination)

print(f"向 {direction} 方向行进 {distance:.1f} 千米")
# 输出: "向 NW 方向行进 21.3 千米"
```

### 示例 3: 查找附近的餐厅

```python
grid = DGGSS2(level=14)

restaurants = [
    {"id": "r1", "name": "Restaurant 1", "lat": 42.36, "lon": -71.05},
    {"id": "r2", "name": "Restaurant 2", "lat": 42.37, "lon": -71.06},
    # ... more restaurants
]

# 用户位置
user_location = (42.365, -71.055)

# 查找 2km 内的餐厅
nearby = grid.find_entities_in_range(
    restaurants,
    center_lat=user_location[0],
    center_lon=user_location[1],
    radius_km=2.0
)

print(f"找到 {len(nearby)} 家餐厅:")
for r in nearby:
    print(f"  {r['name']}: {r['distance_km']:.2f}km, 在{r['direction']}方向")
```

### 示例 4: 构建地理知识图谱

```python
from GraphReasoning.dggs import analyze_entity_relationships
import networkx as nx

entities = [
    {"id": "A", "lat": 42.36, "lon": -71.05, "name": "Location A"},
    {"id": "B", "lat": 42.37, "lon": -71.06, "name": "Location B"},
    {"id": "C", "lat": 42.38, "lon": -71.07, "name": "Location C"},
]

# 分析所有空间关系
relationships = analyze_entity_relationships(entities, level=12)

# 构建知识图谱
KG = nx.MultiDiGraph()

# 添加实体节点
for ent in entities:
    KG.add_node(ent["id"], **ent)

# 添加空间关系边
for rel in relationships:
    # 拓扑关系
    KG.add_edge(
        rel["entity1_id"],
        rel["entity2_id"],
        type="topology",
        relation=rel["topology"]
    )
    
    # 方向关系
    KG.add_edge(
        rel["entity1_id"],
        rel["entity2_id"],
        type="direction",
        direction=rel["direction"],
        distance_km=rel["distance_km"]
    )

# 查询: 找到所有在 A 东北方向的实体
for u, v, data in KG.edges(data=True):
    if u == "A" and data.get("type") == "direction" and data.get("direction") == "NE":
        print(f"{v} 在 A 的东北方向，距离 {data['distance_km']} km")
```

## 拓扑关系说明

| 关系 | 含义 | 使用场景 |
|------|------|----------|
| `equal` | 在同一 cell | 检查是否在同一区域 |
| `contains` | A 包含 B（A 层级更粗） | 城市包含街区 |
| `within` | A 在 B 内（B 层级更粗） | 街区属于城市 |
| `adjacent` | 同层级相邻 | 查找相邻区域 |
| `disjoint` | 不相交 | 完全分离的区域 |

## 方向说明

- **N**: 正北（0°）
- **NE**: 东北（45°）
- **E**: 正东（90°）
- **SE**: 东南（135°）
- **S**: 正南（180°）
- **SW**: 西南（225°）
- **W**: 正西（270°）
- **NW**: 西北（315°）

## 邻近度分类

根据 `distance_threshold_km` 自动分类：
- **near**: < threshold
- **moderate**: threshold ~ 3×threshold
- **distant**: > 3×threshold

## 完整示例

运行以下文件查看完整演示：

```bash
# 基本功能测试
python test_dggs_spatial_relations.py

# 完整示例（包含 8 个场景）
python examples/dggs_examples.py
```

详细文档：`DGGS_HIERARCHICAL.md`
