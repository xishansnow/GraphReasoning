# SSURGO DGGS 离散化 - 快速参考

## 🚀 快速开始

```python
from Dggs.ssurgo import create_ssurgo_sample_data
from Dggs import discretize_ssurgo_agricultural_suitability

# 1. 获取数据
map_units = create_ssurgo_sample_data()

# 2. 离散化 - 评估玉米适宜性
result = discretize_ssurgo_agricultural_suitability(
    map_units, crop='corn', level=12
)

# 3. 查看结果
for cell, data in result.items():
    print(f"{cell}: {data['suitability_class']} ({data['score']}/100)")
```

---

## 📚 API 速查表

### 1. 基础离散化
```python
from Dggs import discretize_ssurgo_map_units

result = discretize_ssurgo_map_units(map_units, level=12, method='centroid')
# method: 'centroid' (快) | 'coverage' (精)
```

### 2. 土壤属性
```python
from Dggs import discretize_ssurgo_soil_properties

result = discretize_ssurgo_soil_properties(
    map_units,
    properties=['pH', 'sand_percent', 'clay_percent'],
    level=12,
    aggregation_funcs={'pH': 'weighted_mean', 'sand_percent': 'mean'}
)
```

### 3. 农业适宜性
```python
from Dggs import discretize_ssurgo_agricultural_suitability

result = discretize_ssurgo_agricultural_suitability(
    map_units, crop='corn', level=12
)
# 可用作物: 'corn', 'wheat', 'soybean', 'alfalfa'
```

### 4. 水文分类
```python
from Dggs import discretize_ssurgo_hydrologic_group

result = discretize_ssurgo_hydrologic_group(map_units, level=12)
# HSG: A (高入渗) -> D (低入渗)
```

---

## 🔧 数据模型

```python
from Dggs.ssurgo import SSURGOMapUnit

mu = SSURGOMapUnit(
    mukey='123456',
    polygon_coords=[(lat1, lon1), (lat2, lon2), ...],
    components=[
        {
            'series_name': 'Inwood',
            'percentage': 70,  # 占比 %
            'pH': 6.8,
            'sand_percent': 25,
            'clay_percent': 35,
            'drainage_class': 'well',
            'hydro_group': 'B'
        }
    ]
)
```

---

## 📊 应用速查

| 需求 | 使用函数 | 输出 |
|-----|--------|------|
| 快速空间索引 | `discretize_ssurgo_map_units` | MUKEY → Cell |
| 土壤成分聚合 | `discretize_ssurgo_soil_properties` | 加权属性 |
| 作物评估 | `discretize_ssurgo_agricultural_suitability` | 适宜性评分 |
| 径流分析 | `discretize_ssurgo_hydrologic_group` | HSG + 入渗率 |
| 土壤层分析 | `discretize_ssurgo_horizon_properties` | 深度特定属性 |

---

## 🎯 常见模式

### 模式1: 农业规划
```python
# 找出所有"高度适合"玉米的地块
suit = discretize_ssurgo_agricultural_suitability(map_units, 'corn', 12)
suitable = {c: d for c, d in suit.items() 
            if d['suitability_class'] == 'Highly Suitable'}
```

### 模式2: 环保评估
```python
# 识别高风险单元 (HSG D - 高径流)
hydro = discretize_ssurgo_hydrologic_group(map_units, 12)
high_risk = {c: d for c, d in hydro.items() 
             if d['primary_hsg'] == 'D'}
```

### 模式3: 知识融合
```python
# 组合多个角度的数据
mu_cells = discretize_ssurgo_map_units(map_units, 12)
props = discretize_ssurgo_soil_properties(map_units, ['pH'], 12)
suit = discretize_ssurgo_agricultural_suitability(map_units, 'corn', 12)

for cell in mu_cells:
    if cell in props and cell in suit:
        combined = {
            'map_unit': mu_cells[cell],
            'soil_pH': props[cell].get('pH_weighted_mean'),
            'corn_suitability': suit[cell]['score']
        }
```

---

## ⚡ 性能提示

| 操作 | 性能 | 优化 |
|-----|------|------|
| Centroid 方法 | 0.5 ms/单元 | 快速查询用 |
| Coverage 方法 | 10-100 ms/单元 | 小区域精确用 |
| 属性聚合 | 2 ms | 批量处理 |
| 适宜性评分 | 1 ms | 缓存结果 |

---

## 📖 完整文档

- [SSURGO_DISCRETIZATION_GUIDE.md](SSURGO_DISCRETIZATION_GUIDE.md) - 详细指南
- [ssurgo_examples.py](ssurgo_examples.py) - 7 个示例
- [DGGS_DISCRETIZATION_GUIDE.md](DGGS_DISCRETIZATION_GUIDE.md) - 基础离散化

---

## 🔗 相关资源

- SSURGO Web: https://websoilsurvey.sc.egov.usda.gov/
- S2 Geometry: https://github.com/google/s2geometry
- DGGS Paper: "The S2 Hierarchical Discrete Global Grid..."
