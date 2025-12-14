# DGGS Value Discretization (DGGS 值离散化)

基于论文 "The S2 Hierarchical Discrete Global Grid as a Nexus for Data Representation, Integration, and Querying Across Geospatial Knowledge Graphs" 实现的离散化方案。

**关键概念**：离散化不仅包括几何位置映射到网格单元，更重要的是包含**数据值的聚合处理**。

## 核心离散化方法

### 1. 直接值分配 (Direct Value Assignment)

将观测值直接分配给包含该观测点的 DGGS 单元。

```python
from Dggs import discretize_direct_assignment

# 气象站观测数据
observations = [
    {"id": "station_A", "lat": 40.7, "lon": -74.0, 
     "temperature": 25.5, "humidity": 65},
    {"id": "station_B", "lat": 40.8, "lon": -74.1, 
     "temperature": 24.2, "humidity": 70}
]

# 离散化：将值分配到单元
result = discretize_direct_assignment(
    observations,
    value_fields=["temperature", "humidity"],
    level=12
)

# 结果: {"89c25a1": {"temperature": 25.5, "humidity": 65, "count": 1}, ...}
```

**适用场景**：稀疏数据、单点观测、精确值保留

---

### 2. 统计聚合 (Statistical Aggregation)

当多个观测值落入同一单元时，使用统计函数聚合（均值、总和、最大值等）。

```python
from Dggs import discretize_aggregate

# 多个传感器可能落入同一单元
sensors = [
    {"lat": 40.7, "lon": -74.0, "temp": 25, "pm25": 35},
    {"lat": 40.7001, "lon": -74.0001, "temp": 26, "pm25": 42}
]

# 使用不同统计函数聚合
result = discretize_aggregate(
    sensors,
    value_fields=["temp", "pm25"],
    level=11,
    agg_funcs={
        "temp": "mean",      # 温度取平均
        "pm25": "max"        # 污染取最大值（最坏情况）
    }
)

# 结果: {"89c25a": {"temp_mean": 25.5, "pm25_max": 42, "count": 2}}
```

**支持的聚合函数**：
- `mean` - 平均值
- `sum` - 总和
- `min` / `max` - 最小/最大值
- `median` - 中位数
- `std` - 标准差
- `count` - 计数
- 自定义函数

**适用场景**：密集观测、传感器网络、统计分析

---

### 3. 多尺度聚合 (Multi-Scale Aggregation)

在多个层级同时聚合数据值，支持跨尺度查询和分析。

```python
from Dggs import discretize_multiscale

# 人口普查数据
census_data = [
    {"lat": 40.7, "lon": -74.0, "population": 5000, "income": 50000},
    # ... 更多数据点
]

# 在多个层级聚合
result = discretize_multiscale(
    census_data,
    value_fields=["population", "income"],
    levels=[8, 10, 12],  # 粗 -> 细
    agg_funcs={
        "population": "sum",
        "income": "mean"
    }
)

# 结果: {
#   8: {"89c2": {"population_sum": 500000, "income_mean": 55000}},
#   10: {"89c25": {"population_sum": 50000, ...}, ...},
#   12: {"89c25a": {...}, ...}
# }
```

**关键特性**：
- 同一数据在不同尺度有不同聚合结果
- 支持层级查询（drill-down/roll-up）
- 实现多分辨率表示

**适用场景**：多尺度分析、自适应详细程度、层级知识图谱

---

### 4. 加权聚合 (Weighted Aggregation)

使用权重聚合（面积加权、人口加权、置信度加权等）。

```python
from Dggs import discretize_weighted_aggregate

# 城市区域数据，按面积加权
districts = [
    {"lat": 40.75, "lon": -73.98, 
     "pollution": 45, "area_sqkm": 2.5, "population": 15000},
    {"lat": 40.751, "lon": -73.981, 
     "pollution": 52, "area_sqkm": 1.8, "population": 12000}
]

# 面积加权聚合
area_weighted = discretize_weighted_aggregate(
    districts,
    value_fields=["pollution"],
    weight_field="area_sqkm",
    level=10,
    agg_func="weighted_mean"
)

# 人口加权聚合
pop_weighted = discretize_weighted_aggregate(
    districts,
    value_fields=["pollution"],
    weight_field="population",
    level=10,
    agg_func="weighted_mean"
)
```

**适用场景**：
- 面积加权统计
- 人口加权指标
- 传感器可靠性加权
- 不同重要性的观测

---

### 5. 插值离散化 (Interpolation-based Discretization)

从稀疏观测点插值到连续网格场。

```python
from Dggs import discretize_interpolate

# 稀疏的温度测站
temp_stations = [
    {"lat": 40.72, "lon": -73.95, "temperature": 25.5},
    {"lat": 40.78, "lon": -74.00, "temperature": 24.2},
    # 仅 4 个测站
]

# 插值到整个区域的网格
interpolated = discretize_interpolate(
    temp_stations,
    value_field="temperature",
    coverage_region=((40.68, -74.05), (40.80, -73.92)),
    level=10,
    interpolation_method="idw",  # 反距离加权
    max_distance_km=15.0,
    power=2.0
)

# 结果：覆盖区域的所有单元都有插值温度值
```

**插值方法**：
- `idw` - 反距离加权（Inverse Distance Weighting）
- `nearest` - 最近邻插值

**适用场景**：连续场表示、空间预测、稀疏数据增强

---

## 完整示例：城市空气质量分析

```python
from Dggs import (
    discretize_direct_assignment,
    discretize_aggregate,
    discretize_multiscale,
    discretize_weighted_aggregate,
    discretize_interpolate
)

# 空气质量监测数据
aq_data = [
    {"id": "monitor_1", "lat": 40.7, "lon": -74.0, 
     "pm25": 35, "no2": 22, "reliability": 0.95},
    # ... 50 个监测站
]

# 1. 加权聚合（按传感器可靠性）
weighted_result = discretize_weighted_aggregate(
    aq_data,
    value_fields=["pm25", "no2"],
    weight_field="reliability",
    level=11,
    agg_func="weighted_mean"
)

# 2. 多尺度分析（健康风险评估用最大值）
multiscale_result = discretize_multiscale(
    aq_data,
    value_fields=["pm25"],
    levels=[9, 11, 13],
    agg_funcs={"pm25": "max"}  # 最坏情况
)

# 3. 插值生成连续污染地图
pollution_map = discretize_interpolate(
    aq_data,
    value_field="pm25",
    coverage_region=((40.65, -74.05), (40.85, -73.95)),
    level=12,
    interpolation_method="idw",
    max_distance_km=5.0
)
```

---

## 与论文的对应关系

| 论文概念 | 实现函数 | 核心功能 |
|---------|---------|---------|
| Direct discretization | `discretize_direct_assignment` | 值直接分配到单元 |
| Aggregation | `discretize_aggregate` | 统计聚合（mean/sum/max等） |
| Multi-resolution | `discretize_multiscale` | 多尺度同时聚合 |
| Weighted aggregation | `discretize_weighted_aggregate` | 加权平均/加权和 |
| Spatial interpolation | `discretize_interpolate` | 空间插值填充 |

---

## 关键区别

### ❌ 错误理解（仅几何）
```python
# 仅映射位置到单元，丢失数据值
result = {"entity_1": "cell_token_abc"}
```

### ✅ 正确理解（几何 + 值）
```python
# 单元包含聚合后的数据值
result = {
    "cell_token_abc": {
        "temperature_mean": 25.5,
        "pollution_max": 45,
        "count": 10,
        "entity_ids": ["sensor_1", "sensor_2", ...]
    }
}
```

---

## 应用场景

1. **环境监测**：温度、污染、降雨等连续场
2. **人口统计**：人口密度、收入、年龄分布等
3. **城市规划**：交通流量、设施密度、用地类型
4. **知识图谱**：地理实体属性的空间聚合
5. **时空分析**：不同时间尺度和空间尺度的综合分析

完整示例见 `discretization_examples.py`。
