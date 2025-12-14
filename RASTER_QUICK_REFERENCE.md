# 通用栅格快速参考
# Generic Raster Quick Reference

**版本**: 2.0  
**模块**: `DGGS/raster.py`

---

## ⚡ 快速开始

### 1. 分类栅格 (土地覆盖、作物类型)
```python
from Dggs import CategoricalPixel, discretize_raster_categorical

# 创建像素
pixels = [
    CategoricalPixel(lat=40.0, lon=-100.0, value=41, category_name="Forest"),
    CategoricalPixel(lat=40.0, lon=-100.01, value=82, category_name="Cropland"),
]

# 离散化
result = discretize_raster_categorical(pixels, level=12)

# 访问结果
for cell_token, data in result.items():
    print(f"主导类别: {data['dominant_category']['name']}")
    print(f"多样性: {data['category_diversity']:.2f}")
```

---

### 2. 连续栅格 (温度、高程、降水)
```python
from Dggs import ContinuousPixel, discretize_raster_continuous

# 创建像素
pixels = [
    ContinuousPixel(lat=40.0, lon=-100.0, value=25.5, unit="celsius"),
    ContinuousPixel(lat=40.0, lon=-100.01, value=26.0, unit="celsius"),
]

# 离散化 (平均值)
result = discretize_raster_continuous(pixels, level=12, aggregation_func='mean')

# 访问结果
for cell_token, data in result.items():
    print(f"平均值: {data['mean']:.2f} {data['unit']}")
    print(f"范围: {data['min']:.2f} - {data['max']:.2f}")
```

---

### 3. 时间序列分析
```python
from Dggs import discretize_raster_temporal

# 准备多年数据
pixels_by_year = {
    '2020': [/* 2020年像素 */],
    '2021': [/* 2021年像素 */]
}

# 离散化
result = discretize_raster_temporal(
    pixels_by_year, 
    level=12, 
    categorical=True  # 分类数据
)
```

---

### 4. 变化检测
```python
from Dggs import calculate_raster_change

# 计算变化
changes = calculate_raster_change(
    before_data=result_2020,
    after_data=result_2021,
    categorical=True  # 分类数据
)

# 检查变化
for cell_token, change in changes.items():
    if change['changed']:
        print(f"{change['transition']}")  # "Forest → Cropland"
```

---

## 📋 数据模型速查

| 类 | 用途 | 示例数据 |
|---|---|---|
| `RasterPixel` | 基类 | 任意栅格像素 |
| `CategoricalPixel` | 分类数据 | NLCD, CDL, MODIS Land Cover |
| `ContinuousPixel` | 连续数据 | PRISM, SRTM, WorldClim, NDVI |

---

## 🎯 函数速查

| 函数 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `discretize_raster_categorical()` | 分类像素 | 主导类别 + 多样性 | 土地覆盖聚合 |
| `discretize_raster_continuous()` | 连续像素 | 统计量 | 气候/地形聚合 |
| `discretize_raster_temporal()` | 时间序列 | 多时间点数据 | 时间序列分析 |
| `calculate_raster_change()` | 前后数据 | 变化检测 | 变化分析 |

---

## 🔧 聚合方法

| 方法 | 说明 | 适用场景 |
|---|---|---|
| `'mean'` | 平均值 | 温度、降水 |
| `'median'` | 中位数 | 抗异常值 |
| `'min'` | 最小值 | 最低温度 |
| `'max'` | 最大值 | 峰值温度 |
| `'sum'` | 总和 | 累积降水 |
| `'custom'` | 自定义函数 | 百分位数等 |

---

## 🌍 DGGS 级别

| Level | 面积 | 适用场景 |
|---|---|---|
| 10 | ~1000 km² | 区域尺度 |
| 11 | ~250 km² | 县级尺度 |
| **12** | **~60 km²** | **农场/流域 (推荐)** |
| 13 | ~15 km² | 田块尺度 |
| 14 | ~4 km² | 精细尺度 |

---

## 💡 常见模式

### 模式 1: NLCD 土地覆盖
```python
nlcd_codes = {11: "Open Water", 41: "Forest", 82: "Cropland"}
pixels = [CategoricalPixel(lat, lon, value=code) for ...]
result = discretize_raster_categorical(pixels, level=12, name_mapping=nlcd_codes)
```

### 模式 2: PRISM 温度
```python
pixels = [ContinuousPixel(lat, lon, value=temp, unit="celsius") for ...]
result = discretize_raster_continuous(pixels, level=12, aggregation_func='mean')
```

### 模式 3: SRTM 高程
```python
pixels = [ContinuousPixel(lat, lon, value=elev, unit="meters") for ...]
result = discretize_raster_continuous(pixels, level=12)
terrain_relief = result[cell]['max'] - result[cell]['min']
```

### 模式 4: 自定义聚合
```python
def percentile_75(values):
    return sorted(values)[int(len(values) * 0.75)]

result = discretize_raster_continuous(
    pixels, 
    level=12, 
    aggregation_func='custom',
    custom_aggregator=percentile_75
)
```

---

## 📊 输出结构

### 分类栅格输出
```python
{
    'cell_token': {
        'total_pixels': 10,
        'total_area_acres': 2.5,
        'categories': {
            'Forest': {'count': 7, 'percent': 70.0},
            'Cropland': {'count': 3, 'percent': 30.0}
        },
        'dominant_category': {'name': 'Forest', 'percent': 70.0},
        'category_diversity': 0.88  # Shannon index
    }
}
```

### 连续栅格输出
```python
{
    'cell_token': {
        'total_pixels': 10,
        'mean': 25.5,
        'median': 25.3,
        'min': 24.0,
        'max': 27.0,
        'std': 0.95,
        'sum': 255.0,
        'unit': 'celsius'
    }
}
```

---

## ✅ 检查清单

使用通用栅格模块前:
- [ ] 确定数据类型 (分类 vs 连续)
- [ ] 选择合适的 DGGS 级别 (推荐 12)
- [ ] 准备 name_mapping (分类数据)
- [ ] 选择聚合方法 (连续数据)
- [ ] 考虑最小像素数阈值

---

## 🔗 相关资源

- **完整文档**: `RASTER_DISCRETIZATION_GUIDE.md`
- **示例代码**: `examples/raster_examples.py`
- **重构总结**: `RASTER_REFACTORING_SUMMARY.md`
- **集成测试**: `test_discretized_to_kg_integration.py`

---

## 🚀 扩展到新栅格类型

```python
from Dggs.discretizer_raster import CategoricalPixel, discretize_raster_categorical

# 1. 定义像素类 (可选 - 也可直接使用 CategoricalPixel)
class MyRasterPixel(CategoricalPixel):
    def __init__(self, lat, lon, code, **kwargs):
        super().__init__(
            lat=lat, lon=lon,
            value=code,
            category_name=MY_MAPPING.get(code),
            **kwargs
        )

# 2. 定义离散化函数 (可选 - 也可直接使用通用函数)
def discretize_my_raster(pixels, level=12):
    return discretize_raster_categorical(pixels, level)
```

---

**最后更新**: 2024  
**维护**: DGGS 开发团队
