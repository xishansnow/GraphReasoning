# 通用栅格离散化指南
# Generic Raster Discretization Guide

**版本**: 2.0  
**文件**: `DGGS/raster.py`  
**作者**: DGGS 开发团队

---

## 📋 概述 (Overview)

`raster.py` 模块提供了**通用的栅格数据离散化框架**，支持任意类型的栅格/网格数据：
- ✅ **分类栅格** (Categorical): 土地覆盖、作物类型、土壤类型
- ✅ **连续栅格** (Continuous): 温度、降水、高程、NDVI、植被覆盖度
- ✅ **时间序列** (Temporal): 多年数据、变化检测
- ✅ **多源数据** (Multi-source): NLCD, CDL, MODIS, PRISM, WorldClim, SRTM, Sentinel

### 🏗️ 架构设计 (Architecture)

```
RasterPixel (基类)
    ├── CategoricalPixel (分类栅格)
    │   └── CDLPixel (CDL 作物数据)
    └── ContinuousPixel (连续栅格)
```

**核心原则**:
- **继承而非重复**: 所有栅格类型从基类派生
- **通用而非特化**: 核心逻辑适用于任何栅格数据
- **可扩展**: 轻松添加新的栅格类型

---

## 🎯 核心功能

### 1. 数据模型 (Data Models)

#### RasterPixel (基类)
```python
from Dggs import RasterPixel

pixel = RasterPixel(
    lat=40.0,
    lon=-100.0,
    value=25.5,
    attributes={'source': 'PRISM', 'quality': 'high'},
    timestamp='2021-01-01'
)
```

**属性**:
- `lat`, `lon`: 像素中心坐标
- `value`: 像素值 (可以是任意类型)
- `attributes`: 元数据字典
- `timestamp`: 时间戳 (可选)

---

#### CategoricalPixel (分类栅格)
```python
from Dggs import CategoricalPixel

pixel = CategoricalPixel(
    lat=40.0,
    lon=-100.0,
    value=41,                        # 类别代码
    category_name="Deciduous Forest", # 类别名称
    category_code=41,                 # 标准代码
    confidence=0.95                   # 分类置信度
)
```

**适用场景**:
- 土地覆盖 (NLCD, ESA CCI, MODIS Land Cover)
- 作物分类 (CDL)
- 土壤类型
- 土地利用

---

#### ContinuousPixel (连续栅格)
```python
from Dggs import ContinuousPixel

pixel = ContinuousPixel(
    lat=40.0,
    lon=-100.0,
    value=25.5,
    unit="celsius",
    precision=0.1,
    quality_flag="good"
)
```

**适用场景**:
- 气候数据 (PRISM, WorldClim, CHIRPS)
- 高程数据 (SRTM, ASTER DEM)
- 遥感指数 (NDVI, EVI, SAVI)
- 植被参数 (LAI, FPAR)

---

### 2. 离散化函数

#### discretize_raster_categorical()
**目的**: 将分类栅格聚合到 DGGS 单元格

```python
from Dggs import discretize_raster_categorical, CategoricalPixel

# 创建像素数据
pixels = [
    CategoricalPixel(lat=40.0, lon=-100.0, value=41, category_name="Forest"),
    CategoricalPixel(lat=40.0, lon=-100.01, value=41, category_name="Forest"),
    CategoricalPixel(lat=40.0, lon=-100.02, value=82, category_name="Cropland"),
]

# 离散化
result = discretize_raster_categorical(
    pixels,
    level=12,              # DGGS 级别
    min_pixels=1,          # 最小像素数
    name_mapping={         # 代码→名称映射 (可选)
        41: "Deciduous Forest",
        82: "Cultivated Crops"
    }
)
```

**输出结构**:
```python
{
    'cell_token_123': {
        'total_pixels': 3,
        'total_area_m2': 900.0,
        'total_area_acres': 0.22,
        'categories': {
            'Forest': {'count': 2, 'percent': 66.67, 'area_acres': 0.15},
            'Cropland': {'count': 1, 'percent': 33.33, 'area_acres': 0.07}
        },
        'dominant_category': {
            'name': 'Forest',
            'code': 41,
            'percent': 66.67,
            'area_acres': 0.15
        },
        'category_diversity': 0.92  # Shannon diversity index
    }
}
```

---

#### discretize_raster_continuous()
**目的**: 将连续栅格聚合到 DGGS 单元格

```python
from Dggs import discretize_raster_continuous, ContinuousPixel

# 创建像素数据
pixels = [
    ContinuousPixel(lat=40.0, lon=-100.0, value=25.5, unit="celsius"),
    ContinuousPixel(lat=40.0, lon=-100.01, value=26.0, unit="celsius"),
    ContinuousPixel(lat=40.0, lon=-100.02, value=24.8, unit="celsius"),
]

# 离散化 - 使用平均值
result = discretize_raster_continuous(
    pixels,
    level=12,
    aggregation_func='mean'  # 'mean', 'median', 'min', 'max', 'sum', 'custom'
)
```

**输出结构**:
```python
{
    'cell_token_123': {
        'total_pixels': 3,
        'mean': 25.43,
        'median': 25.50,
        'min': 24.80,
        'max': 26.00,
        'std': 0.61,
        'sum': 76.30,
        'unit': 'celsius'
    }
}
```

**聚合方法**:
- `'mean'`: 平均值
- `'median'`: 中位数
- `'min'`: 最小值
- `'max'`: 最大值
- `'sum'`: 总和
- `'custom'`: 自定义函数

**自定义聚合示例**:
```python
# 计算第 75 百分位数
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

#### discretize_raster_temporal()
**目的**: 处理时间序列栅格数据

```python
from Dggs import discretize_raster_temporal, CategoricalPixel

# 准备多年数据
pixels_by_time = {
    '2020': [
        CategoricalPixel(lat=40.0, lon=-100.0, value=41, category_name="Forest"),
        # ...更多像素
    ],
    '2021': [
        CategoricalPixel(lat=40.0, lon=-100.0, value=82, category_name="Cropland"),
        # ...更多像素
    ]
}

# 离散化时间序列
result = discretize_raster_temporal(
    pixels_by_time,
    level=12,
    categorical=True  # True for categorical, False for continuous
)
```

**输出结构**:
```python
{
    '2020': {
        'cell_token_123': {/* categorical data */}
    },
    '2021': {
        'cell_token_123': {/* categorical data */}
    }
}
```

---

#### calculate_raster_change()
**目的**: 检测两个时间段之间的变化

```python
from Dggs import calculate_raster_change

# 计算变化 (分类数据)
changes = calculate_raster_change(
    before_data=result_2020,  # 离散化后的数据
    after_data=result_2021,
    categorical=True
)
```

**输出 (分类)**:
```python
{
    'cell_token_123': {
        'changed': True,
        'before': 'Forest',
        'after': 'Cropland',
        'transition': 'Forest → Cropland'
    }
}
```

**输出 (连续)**:
```python
{
    'cell_token_123': {
        'changed': True,
        'before': 25.0,
        'after': 26.5,
        'change_value': 1.5,
        'change_percent': 6.0
    }
}
```

---

## 🌍 应用场景 (Use Cases)

### 场景 1: NLCD 土地覆盖
```python
from Dggs import CategoricalPixel, discretize_raster_categorical

# NLCD 代码映射
nlcd_mapping = {
    11: "Open Water",
    21: "Developed - Open Space",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    81: "Pasture/Hay",
    82: "Cultivated Crops"
}

# 创建像素
pixels = [
    CategoricalPixel(lat=40.0, lon=-100.0, value=41),
    CategoricalPixel(lat=40.0, lon=-100.01, value=42),
    # ...更多像素
]

# 离散化
result = discretize_raster_categorical(
    pixels,
    level=12,
    name_mapping=nlcd_mapping
)
```

---

### 场景 2: PRISM 气候数据
```python
from Dggs import ContinuousPixel, discretize_raster_continuous

# 温度数据
pixels = [
    ContinuousPixel(lat=40.0, lon=-100.0, value=25.5, unit="celsius"),
    ContinuousPixel(lat=40.0, lon=-100.01, value=26.0, unit="celsius"),
    # ...更多像素
]

# 离散化
result = discretize_raster_continuous(
    pixels,
    level=12,
    aggregation_func='mean'
)
```

---

### 场景 3: SRTM 高程数据
```python
from Dggs import ContinuousPixel, discretize_raster_continuous

# 高程数据
pixels = [
    ContinuousPixel(lat=40.0, lon=-105.0, value=2450.5, unit="meters"),
    ContinuousPixel(lat=40.0, lon=-105.01, value=2455.2, unit="meters"),
    # ...更多像素
]

# 离散化 - 计算地形起伏
result = discretize_raster_continuous(
    pixels,
    level=12,
    aggregation_func='mean'
)

# 获取地形起伏
for cell_token, data in result.items():
    relief = data['max'] - data['min']
    print(f"地形起伏: {relief:.1f} meters")
```

---

### 场景 4: 土地覆盖变化检测
```python
from Dggs import discretize_raster_temporal, calculate_raster_change

# 多年数据
pixels_by_year = {
    '2010': [/* 2010年像素 */],
    '2020': [/* 2020年像素 */]
}

# 离散化
result = discretize_raster_temporal(
    pixels_by_year,
    level=12,
    categorical=True
)

# 变化检测
changes = calculate_raster_change(
    result['2010'],
    result['2020'],
    categorical=True
)

# 统计变化
for cell_token, change in changes.items():
    if change['changed']:
        print(f"{change['transition']}")
```

---

## 🔧 与 CDL 模块的集成

CDL 模块已重构为**继承通用栅格模块**:

```python
# CDLPixel 现在继承自 CategoricalPixel
from Dggs import CDLPixel

# 方式 1: 使用 CDL 专用函数 (向后兼容)
from Dggs import discretize_cdl_crop_distribution
result_cdl = discretize_cdl_crop_distribution(cdl_pixels, level=12)

# 方式 2: 使用通用栅格函数 (新方式)
from Dggs import discretize_raster_categorical
result_generic = discretize_raster_categorical(cdl_pixels, level=12)
```

**向后兼容性**: ✅ 完全兼容
- CDL 专用函数仍然可用
- 输出结构保持不变
- 所有现有代码无需修改

---

## 📊 数据流 (Data Flow)

```
原始栅格数据 (GeoTIFF, NetCDF, HDF)
       ↓
像素提取 (RasterPixel / CategoricalPixel / ContinuousPixel)
       ↓
DGGS 离散化 (discretize_raster_*)
       ↓
聚合统计 (分类: 主导类别, 多样性 / 连续: 均值, 标准差)
       ↓
知识图谱转换 (discretized_to_kg.py)
       ↓
RDF 三元组 / NetworkX 图
```

---

## 🎓 最佳实践

### 1. 选择合适的数据模型
- **分类数据** → `CategoricalPixel`
  - 土地覆盖、作物类型、土壤类型
- **连续数据** → `ContinuousPixel`
  - 温度、降水、高程、NDVI

### 2. 选择合适的 DGGS 级别
- **Level 10**: ~1000 km² (区域尺度)
- **Level 11**: ~250 km² (县级尺度)
- **Level 12**: ~60 km² (农场/流域尺度)
- **Level 13**: ~15 km² (田块尺度)
- **Level 14**: ~4 km² (精细尺度)

### 3. 聚合方法选择
- **平均值** (`mean`): 温度、降水
- **中位数** (`median`): 抗异常值
- **最大值** (`max`): 峰值温度、最大降水
- **最小值** (`min`): 最低温度
- **总和** (`sum`): 累积降水、总生物量

### 4. 处理大数据
```python
# 分块处理
def process_large_raster(pixels, chunk_size=10000):
    results = []
    for i in range(0, len(pixels), chunk_size):
        chunk = pixels[i:i+chunk_size]
        result = discretize_raster_categorical(chunk, level=12)
        results.append(result)
    
    # 合并结果
    merged = {}
    for result in results:
        merged.update(result)
    return merged
```

---

## 📚 参考示例

完整示例请参考:
- `examples/raster_examples.py`: 7 个通用栅格示例
- `cdl_examples.py`: CDL 特定示例
- `test_discretized_to_kg_integration.py`: 集成测试

---

## 🔗 相关模块

- `geometry.py`: 几何离散化
- `discretize.py`: 基于值的离散化
- `cdl.py`: CDL 作物数据 (继承 raster.py)
- `ssurgo.py`: SSURGO 土壤数据
- `discretized_to_kg.py`: 知识图谱转换

---

## 📝 扩展新栅格类型

要添加新的栅格类型 (如 MODIS):

```python
from Dggs.discretizer_raster import CategoricalPixel, discretize_raster_categorical

class MODISPixel(CategoricalPixel):
    """MODIS Land Cover pixel"""
    
    def __init__(self, lat, lon, lc_type, year=None, **kwargs):
        # MODIS 代码映射
        lc_mapping = {
            1: "Evergreen Needleleaf Forest",
            2: "Evergreen Broadleaf Forest",
            # ...更多类型
        }
        
        super().__init__(
            lat=lat,
            lon=lon,
            value=lc_type,
            category_name=lc_mapping.get(lc_type, f"Type_{lc_type}"),
            category_code=lc_type,
            timestamp=year,
            **kwargs
        )

# 使用通用函数
def discretize_modis_land_cover(pixels, level=12):
    return discretize_raster_categorical(pixels, level=level)
```

---

## ✅ 总结

通用栅格模块 (`raster.py`) 提供了:
- ✅ **统一接口**: 处理任意栅格数据
- ✅ **灵活扩展**: 轻松添加新数据类型
- ✅ **向后兼容**: 现有代码无需修改
- ✅ **高效聚合**: 支持多种统计方法
- ✅ **时间序列**: 支持变化检测
- ✅ **知识图谱**: 无缝集成 discretized_to_kg

**适用数据源**:
- 土地覆盖: NLCD, CDL, MODIS, ESA CCI
- 气候: PRISM, WorldClim, CHIRPS, Daymet
- 高程: SRTM, ASTER, NED
- 遥感: Landsat, Sentinel, MODIS

**下一步**: 参考 `examples/raster_examples.py` 查看完整示例！
