# SSURGO 数据离散化实现指南

## 📋 概述

**SSURGO (Soil Survey Geographic Database)** 是美国农业部 NRCS 提供的详细土壤调查数据库。本指南介绍如何使用 DGGS 系统将 SSURGO 数据离散化并集成到知识图谱中。

### SSURGO 的关键特点

- **地理精度高**: 最细可到 1:24,000 比例尺
- **多层次数据**: 包含地图单元、土壤成分、土壤学性质
- **属性丰富**: 1000+ 种土壤属性（物理、化学、水文等）
- **广泛覆盖**: 覆盖美国全部州和美国领地

---

## 🏗️ 数据结构

### 核心概念

```
SSURGO Database
├── Map Unit (地图单元) - 代表具有相似土壤的地理区域
│   └── Mukey - 地图单元识别码
├── Component (成分) - 地图单元内的土壤系列
│   ├── Component % - 该成分在地图单元中的百分比
│   └── Soil Properties (土壤属性)
│       ├── Physical - 质地、密度、孔隙度
│       ├── Chemical - pH、有机质、盐分
│       ├── Hydrologic - 渗透速率、保水量
│       └── Horizons (土壤层) - 不同深度的性质
└── Detailed Characteristics (详细特征)
    ├── 排水等级 (Drainage Class)
    ├── 水文土壤分类 (Hydrologic Group)
    ├── 农业适宜性 (Agricultural Suitability)
    └── 工程性质 (Engineering Properties)
```

### 数据模型

```python
from DGGS.ssurgo import SSURGOMapUnit

# 创建地图单元
map_unit = SSURGOMapUnit(
    mukey='123456',  # 地图单元键
    polygon_coords=[(lat1, lon1), (lat2, lon2), ...],  # 多边形坐标
    components=[
        {
            'series_name': 'Inwood',  # 土壤系列名
            'percentage': 70,  # 该成分占比
            'pH': 6.8,
            'sand_percent': 25,
            'clay_percent': 35,
            'drainage_class': 'well',  # 排水类别
            'hydro_group': 'B',  # 水文土壤分类
            'ksat': 0.5  # 饱和导水率
        },
        # ... 其他成分
    ]
)
```

---

## 🔄 离散化方法

### 1. 基础地图单元离散化

**方法**: 将 SSURGO 地图单元映射到 DGGS 单元

```python
from DGGS import discretize_ssurgo_map_units

result = discretize_ssurgo_map_units(
    map_units,
    level=12,  # DGGS 层级
    method='centroid'  # 'centroid' 或 'coverage'
)

# 输出: {cell_token: {'mukey': '123456', 'components': [...], ...}}
```

**两种方法对比**:

| 方法 | 优点 | 缺点 | 用途 |
|-----|-----|-----|-----|
| `centroid` | 快速，简单 | 可能遗漏小区域 | 快速查询，性能关键 |
| `coverage` | 精确，覆盖完整 | 计算量大，慢 | 精确分析，小区域 |

**性能提示**:
- Centroid 方法: ~0.1 ms/单元
- Coverage 方法: ~10-100 ms/单元

---

### 2. 土壤属性聚合

**方法**: 将多个土壤成分的属性加权聚合

```python
from DGGS import discretize_ssurgo_soil_properties

result = discretize_ssurgo_soil_properties(
    map_units,
    properties=['pH', 'sand_percent', 'clay_percent', 'bulk_density'],
    level=12,
    aggregation_funcs={
        'pH': 'weighted_mean',  # 按成分百分比加权
        'sand_percent': 'weighted_mean',
        'clay_percent': 'weighted_mean',
        'bulk_density': 'weighted_mean'
    },
    weight_by_component=True
)

# 输出: {cell_token: {'pH_weighted_mean': 6.8, 'sand_percent_weighted_mean': 25, ...}}
```

**支持的聚合函数**:
- `mean` - 简单平均
- `weighted_mean` - 按成分百分比加权平均 ⭐
- `sum` - 求和
- `max` - 最大值
- `min` - 最小值

**推荐做法**: 使用 `weighted_mean` 因为土壤成分有不同的占比

---

### 3. 农业适宜性评估

**方法**: 基于土壤属性计算作物适宜性评分

```python
from DGGS import discretize_ssurgo_agricultural_suitability

result = discretize_ssurgo_agricultural_suitability(
    map_units,
    crop='corn',  # 'corn', 'wheat', 'soybean', 'alfalfa'
    level=12
)

# 输出: {
#   cell_token: {
#     'crop': 'corn',
#     'suitability_class': 'Highly Suitable',
#     'score': 95.5,  # 0-100
#     'dominant_series': 'Inwood',
#     'dominant_component_pct': 70
#   }
# }
```

**适宜性等级**:
- `Highly Suitable` (80-100): 适合该作物
- `Suitable` (60-79): 可以种植
- `Marginally Suitable` (40-59): 需要改良
- `Not Suitable` (<40): 不适合

---

### 4. 水文土壤分类

**方法**: 离散化 USDA 水文土壤分类 (HSG) 用于径流分析

```python
from DGGS import discretize_ssurgo_hydrologic_group

result = discretize_ssurgo_hydrologic_group(map_units, level=12)

# 输出: {
#   cell_token: {
#     'hydro_group': 'B',
#     'infiltration_in_hr': 0.25,  # 英寸/小时
#     'primary_hsg': 'B'
#   }
# }
```

**USDA 水文土壤分类**:

| 分类 | 入渗速率 | 特征 | 径流潜力 |
|-----|--------|------|--------|
| A | > 0.8 in/hr | 砂质，入渗快 | 低 |
| B | 0.25-0.8 | 壤土到砂壤土 | 中-低 |
| C | 0.1-0.25 | 砂粘土到粘壤土 | 中-高 |
| D | < 0.05 | 粘质，入渗慢 | 高 |

---

### 5. 土壤层次分析

**方法**: 分析不同深度的土壤属性

```python
from DGGS.ssurgo import discretize_ssurgo_horizon_properties

result = discretize_ssurgo_horizon_properties(
    map_units,
    horizon_depths={
        'A': (0, 25),      # 表层
        'B': (25, 100),    # 心土
        'C': (100, 200)    # 母质
    },
    properties=['clay_percent', 'bulk_density'],
    level=12
)

# 输出: {
#   cell_token: {
#     'A': {'clay_percent': 20, 'bulk_density': 1.4},
#     'B': {'clay_percent': 35, 'bulk_density': 1.5},
#     'C': {'clay_percent': 15, 'bulk_density': 1.6}
#   }
# }
```

---

## 📊 使用场景

### 场景1: 农业规划

```python
# 评估哪些地区适合种植特定作物
map_units = load_ssurgo_data()

corn_suitability = discretize_ssurgo_agricultural_suitability(
    map_units, crop='corn', level=12
)

# 找出所有"高度适合"的单元格
suitable_cells = [
    cell for cell, data in corn_suitability.items()
    if data['suitability_class'] == 'Highly Suitable'
]
```

### 场景2: 环境影响评估

```python
# 分析污染物渗透风险
hydro_groups = discretize_ssurgo_hydrologic_group(map_units, level=12)

# 分类 D 类土壤（高风险）
high_risk_cells = [
    cell for cell, data in hydro_groups.items()
    if data['primary_hsg'] == 'D'
]
```

### 场景3: 知识图谱集成

```python
# 构建土壤知识图谱
map_units_cells = discretize_ssurgo_map_units(map_units, level=12)
properties_cells = discretize_ssurgo_soil_properties(
    map_units,
    properties=['pH', 'sand_percent', 'clay_percent'],
    level=12
)
suitability_cells = discretize_ssurgo_agricultural_suitability(
    map_units, crop='corn', level=12
)

# 融合多个角度的数据
for cell in map_units_cells:
    if cell in properties_cells and cell in suitability_cells:
        combined_data = {
            'map_unit': map_units_cells[cell],
            'properties': properties_cells[cell],
            'suitability': suitability_cells[cell]
        }
        # 添加到知识图谱
```

---

## 🛠️ 实现要点

### 1. 数据导入

```python
# 从 CSV 导入
from DGGS.ssurgo import parse_ssurgo_csv

map_units = parse_ssurgo_csv('ssurgo_data.csv')

# 或手动创建
from DGGS.ssurgo import SSURGOMapUnit

map_unit = SSURGOMapUnit(
    mukey='123456',
    polygon_coords=[(40.7, -74.0), (40.8, -74.1), ...],
    components=[...]
)
```

### 2. 多尺度分析

```python
# 在不同 DGGS 层级分析
for level in [10, 12, 14]:
    result = discretize_ssurgo_map_units(map_units, level=level)
    print(f"Level {level}: {len(result)} cells")

# 更粗的层级 → 汇总统计
# 更细的层级 → 详细特征
```

### 3. 性能优化

```python
# ✅ 快速查询 - 使用 centroid
quick_result = discretize_ssurgo_map_units(
    map_units, level=12, method='centroid'
)

# ✅ 精确分析 - 批量处理小区域
small_units = [mu for mu in map_units if area < 10_km2]
precise_result = discretize_ssurgo_map_units(
    small_units, level=13, method='coverage'
)
```

---

## 📚 示例

完整示例见 `ssurgo_examples.py`:

1. **基础地图单元离散化** - 将 SSURGO 单元映射到格子
2. **土壤属性聚合** - 加权平均土壤属性
3. **农业适宜性** - 作物种植评估
4. **水文分类** - 径流和入渗分析
5. **多尺度分析** - 跨分辨率统计
6. **质地分类** - USDA 土壤质地三角
7. **知识图谱集成** - 多角度数据融合

运行示例:
```bash
python3 ssurgo_examples.py
```

---

## 🔗 与论文的关联

本实现基于论文关键思想：

> **"The S2 Hierarchical Discrete Global Grid as a Nexus for Data Representation, Integration, and Querying Across Geospatial Knowledge Graphs"**

- **数据表示**: SSURGO 地理数据 → DGGS 离散化
- **多源集成**: 几何 + 属性值聚合
- **知识图谱**: 土壤属性、农业适宜性、水文特征 → 联系节点
- **多尺度查询**: 层级式 DGGS 支持自适应详细程度

---

## ✅ 性能指标

使用 2 个地图单元，1 级 12 的 DGGS:

| 操作 | 时间 | 内存 |
|-----|------|------|
| 基础离散化 (centroid) | 0.5 ms | < 1 MB |
| 属性聚合 (5 属性) | 2 ms | 1-2 MB |
| 农业适宜性 | 1 ms | < 1 MB |
| 水文分类 | 0.5 ms | < 1 MB |

---

## 📖 相关资源

- [SSURGO 官网](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/)
- [Web Soil Survey](https://websoilsurvey.sc.egov.usda.gov/)
- [DGGS 值离散化指南](DGGS_DISCRETIZATION_GUIDE.md)
- [DGGS 模块结构](DGGS_MODULE_STRUCTURE.md)
