"""
通用栅格离散化示例与 CDL 案例研究
Generic Raster Discretization Examples with CDL Case Study

演示如何使用通用的 raster.py 模块处理不同类型的栅格数据：

1. 通用栅格示例:
   - 分类栅格 (Categorical Raster): 土地覆盖 NLCD 数据
   - 连续栅格 (Continuous Raster): 温度、降水、高程
   - 时间序列栅格 (Temporal Raster): 多年数据变化
   - 变化检测 (Change Detection): 前后对比分析

2. CDL 作为通用栅格模块的案例研究:
   - CDL (Cropland Data Layer) 是 USDA NASS 提供的作物覆盖栅格数据
   - 30m 分辨率，年度更新，254+ 种作物类型
   - 演示如何使用通用栅格 API 处理特定领域数据
   - 案例包括: 作物分布、作物轮作模式、农业强度分析

3. 架构设计:
   - CDLPixel 继承 CategoricalPixel，可使用通用栅格 API
   - CDL 专用函数（如 discretize_cdl_crop_distribution）可使用通用函数实现
   - 演示 "域模块通过扩展通用模块来解决特定问题" 的设计模式
"""

from pathlib import Path
import sys

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from DGGS import (
    RasterPixel,
    CategoricalPixel,
    ContinuousPixel,
    discretize_raster_categorical,
    discretize_raster_continuous,
    discretize_raster_temporal,
    calculate_raster_change,
    DGGSS2,
    SpatialEntity,
)
from typing import Dict, List, Sequence, Any, Optional, Tuple
import statistics


####################################################################
# CDL Constants (USDA Crop Codes)
####################################################################

# 美国农业部 CDL 作物代码映射表 (254+ 种作物类型)
CDL_CROP_CODES = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet corn",
    13: "Popcorn",
    14: "Mint",
    21: "Barley",
    22: "Durum wheat",
    23: "Spring wheat",
    24: "Winter wheat",
    25: "Other small grains",
    26: "Double crop winter wheat/soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rapeseed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other hay/non-alfalfa",
    38: "Catnip",
    41: "Sugarbeets",
    42: "Dry beans",
    43: "Potatoes",
    44: "Other crops",
    45: "Sugarcane",
    46: "Sweet potatoes",
    47: "Miscellaneous vegetables & fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chickpeas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/wildflowers",
    59: "Sod/grass seed",
    60: "Switchgrass",
    61: "Fallow/idle cropland",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    81: "Pasture/grass",
    82: "Forest",
    83: "Urban/developed",
    87: "Wetlands",
    88: "Water",
    92: "Aquaculture",
    111: "Open water",
    112: "Perennial ice/snow",
    121: "Developed/open space",
    122: "Developed/low intensity",
    123: "Developed/medium intensity",
    124: "Developed/high intensity",
    131: "Barren",
    141: "Deciduous forest",
    142: "Evergreen forest",
    143: "Mixed forest",
    152: "Shrubland",
    161: "Pasture/hay",
    165: "Winter wheat",
    181: "Pasture/grass",
    182: "Forest",
    190: "Woody wetlands",
    195: "Herbaceous wetlands",
}

# 作物分类（用于高层次分析）
CDL_CROP_CATEGORIES = {
    'cereal_grains': [1, 4, 21, 22, 23, 24, 25, 27, 28, 29, 30],
    'oilseeds': [5, 6, 31, 32, 33, 34, 35],
    'legumes': [36, 42, 51, 52, 53],
    'specialty_crops': [12, 13, 14, 38, 41, 48, 49, 50, 54, 55, 56, 57],
    'vegetables': [47, 48, 49, 50, 54],
    'pasture_hay': [36, 37, 59, 81, 161],
    'non_crop': [61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 112, 121, 122, 123, 124, 131, 141, 142, 143, 152, 165, 181, 182, 190, 195],
}


####################################################################
# CDL 像素类和辅助函数（用于示例）
####################################################################

class CDLPixel(CategoricalPixel):
    """CDL 栅格像素类 - 继承自 CategoricalPixel
    
    这是一个示例，展示如何扩展通用的 CategoricalPixel 类来处理特定领域的数据。
    """
    
    def __init__(self, lat: float, lon: float, crop_code: int, year: int, 
                 confidence: Optional[float] = None, pixel_area_m2: float = 900.0):
        """
        Args:
            lat: 纬度
            lon: 经度
            crop_code: USDA CDL 作物代码 (1-254)
            year: 数据年份
            confidence: 置信度分数 (0-100)
            pixel_area_m2: 像素面积（平方米），默认 30m x 30m = 900 m²
        """
        crop_name = CDL_CROP_CODES.get(crop_code, f"Unknown ({crop_code})")
        
        super().__init__(
            lat=lat,
            lon=lon,
            value=crop_code,
            category_name=crop_name,
            category_code=crop_code,
            confidence=confidence if confidence is not None else 100.0,
            attributes={
                'year': year,
                'pixel_area_m2': pixel_area_m2
            },
            timestamp=str(year)
        )
        
        self.crop_code = crop_code
        self.year = year
        self.crop_name = crop_name
        self.pixel_area_m2 = pixel_area_m2
    
    def get_crop_category(self) -> str:
        """获取作物的高层次分类"""
        for category, codes in CDL_CROP_CATEGORIES.items():
            if self.crop_code in codes:
                return category
        return "other"
    
    def is_agricultural(self) -> bool:
        """判断是否为农业用地（vs 非作物用地）"""
        return self.get_crop_category() != "non_crop"


def discretize_cdl_crop_distribution(
    pixels: Sequence[CDLPixel],
    level: int = 12,
    min_pixels: int = 1
) -> Dict[str, Dict[str, Any]]:
    """CDL 作物分布离散化 - 使用通用栅格 API 的示例
    
    这个函数展示如何使用通用的 discretize_raster_categorical 来处理 CDL 数据。
    """
    result = discretize_raster_categorical(
        pixels,
        level=level,
        min_pixels=min_pixels,
        value_attr='crop_code',
        name_mapping=CDL_CROP_CODES
    )
    
    # 重命名字段以符合 CDL 术语
    final_result = {}
    for cell_token, data in result.items():
        final_result[cell_token] = {
            'total_pixels': data['total_pixels'],
            'total_area_m2': data['total_area_m2'],
            'total_area_acres': data['total_area_acres'],
            'crops': data['categories'],
            'dominant_crop': data['dominant_category'],
            'crop_diversity': data['category_diversity'],
            'year': data.get('timestamp')
        }
    
    return final_result


def discretize_cdl_crop_categories(
    pixels: Sequence[CDLPixel],
    level: int = 12
) -> Dict[str, Dict[str, Any]]:
    """按作物类别分组的离散化"""
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for pixel in pixels:
        cell_token = grid.latlon_to_token(pixel.lat, pixel.lon, level)
        
        if cell_token not in result:
            result[cell_token] = {
                'total_pixels': 0,
                'category_counts': {}
            }
        
        result[cell_token]['total_pixels'] += 1
        category = pixel.get_crop_category()
        
        if category not in result[cell_token]['category_counts']:
            result[cell_token]['category_counts'][category] = 0
        
        result[cell_token]['category_counts'][category] += 1
    
    # 转换为百分比
    for cell_token, data in result.items():
        total = data['total_pixels']
        categories = {}
        max_count = 0
        dominant_category = None
        ag_pixels = 0
        
        for category, count in data['category_counts'].items():
            percent = (count / total) * 100
            categories[category] = {
                'count': count,
                'percent': percent
            }
            
            if count > max_count:
                max_count = count
                dominant_category = {'name': category, 'percent': percent}
            
            if category != 'non_crop':
                ag_pixels += count
        
        result[cell_token] = {
            'total_pixels': total,
            'categories': categories,
            'dominant_category': dominant_category,
            'is_agricultural': ag_pixels / total > 0.5,
            'agricultural_percent': (ag_pixels / total) * 100
        }
    
    return result


def discretize_cdl_rotation_patterns(
    pixels_by_year: Dict[int, Sequence[CDLPixel]],
    level: int = 12,
    min_years: int = 2
) -> Dict[str, Dict[str, Any]]:
    """检测作物轮作模式"""
    # 首先分析每年的作物分布
    result: Dict[str, Dict[int, Any]] = {}
    
    for year, pixels in pixels_by_year.items():
        crop_dist = discretize_cdl_crop_distribution(pixels, level=level)
        
        for cell_token, data in crop_dist.items():
            if cell_token not in result:
                result[cell_token] = {}
            
            result[cell_token][year] = {
                'dominant_crop': data['dominant_crop']['name'],
                'crop_diversity': data['crop_diversity'],
            }
    
    # 分析轮作模式
    rotation_result: Dict[str, Dict[str, Any]] = {}
    
    for cell_token, year_data in result.items():
        if len(year_data) < min_years:
            continue
        
        years = sorted(year_data.keys())
        sequence = [year_data[year]['dominant_crop'] for year in years]
        
        # 计算转换次数
        transitions: Dict[str, Dict[str, int]] = {}
        for i in range(len(sequence) - 1):
            from_crop = sequence[i]
            to_crop = sequence[i + 1]
            
            if from_crop not in transitions:
                transitions[from_crop] = {}
            if to_crop not in transitions[from_crop]:
                transitions[from_crop][to_crop] = 0
            
            transitions[from_crop][to_crop] += 1
        
        # 计算规律性
        if len(sequence) > 1:
            most_common_transition = max(
                (count for counts in transitions.values() for count in counts.values()),
                default=0
            )
            regularity = most_common_transition / (len(sequence) - 1)
        else:
            regularity = 0
        
        rotation_result[cell_token] = {
            'rotation_sequence': sequence,
            'pattern_regularity': min(regularity, 1.0),
            'num_years': len(years),
            'transitions': {k: dict(v) for k, v in transitions.items()}
        }
    
    return rotation_result


def discretize_cdl_agricultural_intensity(
    pixels: Sequence[CDLPixel],
    level: int = 12
) -> Dict[str, Dict[str, Any]]:
    """评估农业强度"""
    crop_dist = discretize_cdl_crop_distribution(pixels, level=level)
    categories = discretize_cdl_crop_categories(pixels, level=level)
    
    result: Dict[str, Dict[str, Any]] = {}
    
    for cell_token in crop_dist.keys():
        if cell_token not in categories:
            continue
        
        crop_data = crop_dist[cell_token]
        cat_data = categories[cell_token]
        
        diversity = crop_data['crop_diversity']
        dominant_pct = crop_data['dominant_crop']['percent']
        
        is_monoculture = dominant_pct > 80
        
        # 强度评分: 0-100
        intensity_score = (
            (1 - diversity) * 50 +
            (dominant_pct - 50) * 0.5 +
            (cat_data['agricultural_percent'] - 50) * 0.5
        )
        intensity_score = max(0, min(100, intensity_score))
        
        if intensity_score > 70:
            intensity = 'intensive'
        elif intensity_score > 40:
            intensity = 'moderate'
        else:
            intensity = 'extensive'
        
        result[cell_token] = {
            'intensity': intensity,
            'intensity_score': intensity_score,
            'ag_percent': cat_data['agricultural_percent'],
            'monoculture': is_monoculture,
            'dominant_crop': crop_data['dominant_crop']['name'],
            'crop_diversity': diversity
        }
    
    return result


####################################################################
# CDL 工具函数 - 用于测试和数据准备
####################################################################

def create_cdl_sample_data(year: int = 2021) -> List[CDLPixel]:
    """
    创建 CDL 样例数据 - 用于测试和演示
    
    参数:
        year: 年份，默认 2021
        
    返回:
        CDL 像素列表
    """
    # 生成不同作物类型的样例数据
    sample_pixels = [
        # 玉米 (corn)
        CDLPixel(lat=40.0, lon=-100.0, crop_code=1, year=year, confidence=0.95),
        CDLPixel(lat=40.0001, lon=-100.0, crop_code=1, year=year, confidence=0.93),
        CDLPixel(lat=40.0002, lon=-100.0, crop_code=1, year=year, confidence=0.92),
        
        # 大豆 (soybeans)
        CDLPixel(lat=40.0, lon=-100.01, crop_code=5, year=year, confidence=0.94),
        CDLPixel(lat=40.0001, lon=-100.01, crop_code=5, year=year, confidence=0.96),
        
        # 冬小麦 (winter wheat)
        CDLPixel(lat=40.0, lon=-100.02, crop_code=24, year=year, confidence=0.91),
        CDLPixel(lat=40.0001, lon=-100.02, crop_code=24, year=year, confidence=0.90),
        
        # 苜蓿 (alfalfa)
        CDLPixel(lat=40.0, lon=-100.03, crop_code=36, year=year, confidence=0.88),
        
        # 森林 (forest)
        CDLPixel(lat=40.001, lon=-100.0, crop_code=63, year=year, confidence=0.97),
        CDLPixel(lat=40.001, lon=-100.01, crop_code=63, year=year, confidence=0.96),
        
        # 草地 (grassland)
        CDLPixel(lat=40.001, lon=-100.02, crop_code=176, year=year, confidence=0.92),
        
        # 开发用地 (developed)
        CDLPixel(lat=40.001, lon=-100.03, crop_code=121, year=year, confidence=0.99),
    ]
    
    return sample_pixels


def parse_cdl_csv(csv_file_path: str, year: Optional[int] = None) -> List[CDLPixel]:
    """
    从 CSV 文件解析 CDL 数据
    
    CSV 格式应包含列: lat, lon, crop_code, (可选: year, confidence)
    
    参数:
        csv_file_path: CSV 文件路径
        year: 年份，如果 CSV 中没有年份列则使用此值
        
    返回:
        CDL 像素列表
    """
    import csv
    
    pixels = []
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row['lat'])
            lon = float(row['lon'])
            crop_code = int(row['crop_code'])
            
            # 年份：优先使用 CSV 中的，否则使用参数
            pixel_year = int(row.get('year', year or 2021))
            
            # 置信度：可选
            confidence = float(row['confidence']) if 'confidence' in row else None
            
            pixels.append(CDLPixel(
                lat=lat,
                lon=lon,
                crop_code=crop_code,
                year=pixel_year,
                confidence=confidence
            ))
    
    return pixels


####################################################################
# CDL 知识图谱转换函数
####################################################################

def discretized_cdl_to_triplets(cell_token: str, cdl_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """将离散化的 CDL 单元格数据转换为 RDF 三元组
    
    用于将 CDL 离散化结果转换为知识图谱三元组格式。
    
    Args:
        cell_token: DGGS 单元格标识符 (如 '89c25a3')
        cdl_data: discretize_cdl_crop_distribution() 的输出
    
    Returns:
        (subject, predicate, object) 三元组列表
        
    Example:
        cdl_pixels = create_cdl_sample_data(year=2021)
        result = discretize_cdl_crop_distribution(cdl_pixels, level=12)
        
        for cell_token, cdl_data in result.items():
            triplets = discretized_cdl_to_triplets(cell_token, cdl_data)
            for s, p, o in triplets:
                print(f"{s} --[{p}]--> {o}")
    """
    triplets = []
    
    # 单元格实体
    entity = SpatialEntity(
        f'cdl_{cell_token}',
        'CDLCell',
        {
            'dggs_level': 12,
            'total_pixels': cdl_data.get('total_pixels'),
            'total_area_acres': round(cdl_data.get('total_area_acres', 0), 2),
        }
    )
    triplets.extend(entity.to_triplets())
    
    # 主导作物关系
    dom_crop = cdl_data.get('dominant_crop', {})
    if dom_crop:
        triplets.append((
            f'cdl_{cell_token}',
            'has_dominant_crop',
            f'crop_{dom_crop.get("code", "unknown")}'
        ))
        triplets.append((
            f'crop_{dom_crop.get("code")}',
            'rdf:type',
            'CropType'
        ))
        triplets.append((
            f'crop_{dom_crop.get("code")}',
            'name',
            dom_crop.get('name', 'Unknown')
        ))
    
    # 作物组成
    crops = cdl_data.get('crops', {})
    for crop_name, crop_info in crops.items():
        crop_id = crop_name.lower().replace(' ', '_').replace('/', '_')
        triplets.append((
            f'cdl_{cell_token}',
            'contains_crop',
            f'crop_{crop_id}'
        ))
        triplets.append((
            f'crop_{crop_id}',
            'percentage',
            str(round(crop_info.get('percent', 0), 1))
        ))
        triplets.append((
            f'crop_{crop_id}',
            'area_acres',
            str(round(crop_info.get('area_acres', 0), 2))
        ))
    
    # 多样性指标
    diversity = cdl_data.get('crop_diversity')
    if diversity is not None:
        triplets.append((
            f'cdl_{cell_token}',
            'crop_diversity',
            str(round(abs(diversity), 2))
        ))
    
    return triplets


def discretized_agricultural_intensity_to_triplets(cell_token: str, intensity_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """将农业强度评估结果转换为 RDF 三元组
    
    Args:
        cell_token: DGGS 单元格标识符
        intensity_data: discretize_cdl_agricultural_intensity() 的输出
    
    Returns:
        RDF 三元组列表
    """
    triplets = []
    
    entity = SpatialEntity(
        f'ag_intensity_{cell_token}',
        'AgriculturalIntensity',
        {
            'intensity_level': intensity_data.get('intensity'),
            'intensity_score': intensity_data.get('intensity_score'),
            'is_monoculture': intensity_data.get('monoculture'),
            'agricultural_percent': intensity_data.get('ag_percent'),
        }
    )
    triplets.extend(entity.to_triplets())
    
    # 关系
    intensity = intensity_data.get('intensity')
    if intensity:
        triplets.append((
            f'ag_intensity_{cell_token}',
            'has_intensity_category',
            intensity
        ))
    
    return triplets


####################################################################
# 示例 1-5: 通用栅格离散化示例
####################################################################

def example_1_categorical_land_cover():
    """示例 1: 分类栅格 - 土地覆盖数据 (NLCD)"""
    print("\n" + "="*70)
    print("示例 1: 分类栅格 - 土地覆盖离散化 (NLCD)")
    print("="*70)
    
    # 创建土地覆盖像素数据 (模拟 NLCD 数据)
    land_cover_codes = {
        11: "Open Water",
        21: "Developed - Open Space",
        22: "Developed - Low Intensity",
        41: "Deciduous Forest",
        42: "Evergreen Forest",
        81: "Pasture/Hay",
        82: "Cultivated Crops"
    }
    
    pixels = [
        CategoricalPixel(lat=40.0, lon=-100.0, value=41, category_name="Deciduous Forest"),
        CategoricalPixel(lat=40.0, lon=-100.01, value=41, category_name="Deciduous Forest"),
        CategoricalPixel(lat=40.0, lon=-100.02, value=42, category_name="Evergreen Forest"),
        CategoricalPixel(lat=40.0, lon=-100.03, value=81, category_name="Pasture/Hay"),
        CategoricalPixel(lat=40.0, lon=-100.04, value=82, category_name="Cultivated Crops"),
        CategoricalPixel(lat=40.001, lon=-100.0, value=41, category_name="Deciduous Forest"),
        CategoricalPixel(lat=40.001, lon=-100.01, value=81, category_name="Pasture/Hay"),
        CategoricalPixel(lat=40.001, lon=-100.02, value=82, category_name="Cultivated Crops"),
    ]
    
    # 离散化
    result = discretize_raster_categorical(
        pixels,
        level=12,
        min_pixels=1,
        name_mapping=land_cover_codes
    )
    
    print(f"\n✅ 离散化了 {len(result)} 个 DGGS 单元格")
    
    # 显示结果
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  总像素: {data['total_pixels']}")
        print(f"  面积: {data['total_area_acres']:.2f} acres")
        print(f"  主导类型: {data['dominant_category']['name']} ({data['dominant_category']['percent']:.1f}%)")
        print(f"  多样性指数: {data['category_diversity']:.2f}")
        print(f"  类别分布:")
        for cat_name, cat_info in data['categories'].items():
            print(f"    - {cat_name}: {cat_info['percent']:.1f}% ({cat_info['count']} pixels)")


def example_2_continuous_temperature():
    """示例 2: 连续栅格 - 温度数据"""
    print("\n" + "="*70)
    print("示例 2: 连续栅格 - 温度数据离散化 (PRISM)")
    print("="*70)
    
    # 创建温度像素数据 (模拟 PRISM 温度数据)
    pixels = [
        ContinuousPixel(lat=40.0, lon=-100.0, value=25.5, unit="celsius"),
        ContinuousPixel(lat=40.0, lon=-100.01, value=26.0, unit="celsius"),
        ContinuousPixel(lat=40.0, lon=-100.02, value=24.8, unit="celsius"),
        ContinuousPixel(lat=40.001, lon=-100.0, value=25.2, unit="celsius"),
        ContinuousPixel(lat=40.001, lon=-100.01, value=26.5, unit="celsius"),
        ContinuousPixel(lat=40.002, lon=-100.0, value=23.5, unit="celsius"),
        ContinuousPixel(lat=40.002, lon=-100.01, value=24.0, unit="celsius"),
    ]
    
    # 离散化 - 使用平均值
    result_mean = discretize_raster_continuous(
        pixels,
        level=12,
        aggregation_func='mean'
    )
    
    print(f"\n✅ 离散化了 {len(result_mean)} 个 DGGS 单元格")
    
    # 显示结果
    for cell_token, data in result_mean.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  像素数: {data['total_pixels']}")
        print(f"  平均温度: {data['mean']:.2f} {data['unit']}")
        print(f"  温度范围: {data['min']:.2f} - {data['max']:.2f} {data['unit']}")
        print(f"  标准差: {data['std']:.2f} {data['unit']}")
        print(f"  中位数: {data['median']:.2f} {data['unit']}")


def example_3_continuous_elevation():
    """示例 3: 连续栅格 - 高程数据"""
    print("\n" + "="*70)
    print("示例 3: 连续栅格 - 高程数据离散化 (SRTM DEM)")
    print("="*70)
    
    # 创建高程像素数据 (模拟 SRTM DEM)
    pixels = [
        ContinuousPixel(lat=40.0, lon=-105.0, value=2450.5, unit="meters"),
        ContinuousPixel(lat=40.0, lon=-105.01, value=2455.2, unit="meters"),
        ContinuousPixel(lat=40.0, lon=-105.02, value=2448.8, unit="meters"),
        ContinuousPixel(lat=40.001, lon=-105.0, value=2460.1, unit="meters"),
        ContinuousPixel(lat=40.001, lon=-105.01, value=2465.5, unit="meters"),
    ]
    
    # 离散化 - 使用多种聚合方法
    result = discretize_raster_continuous(
        pixels,
        level=12,
        aggregation_func='mean'
    )
    
    print(f"\n✅ 离散化了 {len(result)} 个 DGGS 单元格")
    
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  平均高程: {data['mean']:.1f} {data['unit']}")
        print(f"  最大高程: {data['max']:.1f} {data['unit']}")
        print(f"  最小高程: {data['min']:.1f} {data['unit']}")
        print(f"  高程差 (地形起伏): {data['max'] - data['min']:.1f} {data['unit']}")


def example_4_temporal_series():
    """示例 4: 时间序列栅格 - 多年土地覆盖变化"""
    print("\n" + "="*70)
    print("示例 4: 时间序列栅格 - 多年土地覆盖变化")
    print("="*70)
    
    # 2020 年土地覆盖
    pixels_2020 = [
        CategoricalPixel(lat=40.0, lon=-100.0, value=41, category_name="Forest", timestamp="2020"),
        CategoricalPixel(lat=40.0, lon=-100.01, value=41, category_name="Forest", timestamp="2020"),
        CategoricalPixel(lat=40.0, lon=-100.02, value=81, category_name="Pasture", timestamp="2020"),
    ]
    
    # 2021 年土地覆盖 (部分转换为农田)
    pixels_2021 = [
        CategoricalPixel(lat=40.0, lon=-100.0, value=41, category_name="Forest", timestamp="2021"),
        CategoricalPixel(lat=40.0, lon=-100.01, value=82, category_name="Cropland", timestamp="2021"),
        CategoricalPixel(lat=40.0, lon=-100.02, value=82, category_name="Cropland", timestamp="2021"),
    ]
    
    # 离散化时间序列
    result = discretize_raster_temporal(
        pixels_by_time={'2020': pixels_2020, '2021': pixels_2021},
        level=12,
        categorical=True
    )
    
    print(f"\n✅ 处理了 {len(result)} 个时间步")
    
    for year, cells in result.items():
        print(f"\n年份: {year}")
        for cell_token, data in cells.items():
            print(f"  单元格 {cell_token}: {data['dominant_category']['name']}")
    
    # 计算变化
    if '2020' in result and '2021' in result:
        changes = calculate_raster_change(
            result['2020'],
            result['2021'],
            categorical=True
        )
        
        print(f"\n变化检测:")
        for cell_token, change_data in changes.items():
            if change_data['changed']:
                print(f"  单元格 {cell_token}: {change_data['before']} → {change_data['after']}")


def example_5_temperature_change():
    """示例 5: 连续栅格变化 - 温度变化检测"""
    print("\n" + "="*70)
    print("示例 5: 连续栅格变化 - 温度变化检测")
    print("="*70)
    
    # 2020 年温度
    pixels_2020 = [
        ContinuousPixel(lat=40.0, lon=-100.0, value=25.0, unit="celsius", timestamp="2020"),
        ContinuousPixel(lat=40.0, lon=-100.01, value=24.5, unit="celsius", timestamp="2020"),
    ]
    
    # 2021 年温度 (升温)
    pixels_2021 = [
        ContinuousPixel(lat=40.0, lon=-100.0, value=26.5, unit="celsius", timestamp="2021"),
        ContinuousPixel(lat=40.0, lon=-100.01, value=25.8, unit="celsius", timestamp="2021"),
    ]
    
    # 离散化
    result_2020 = discretize_raster_continuous(pixels_2020, level=12)
    result_2021 = discretize_raster_continuous(pixels_2021, level=12)
    
    # 计算变化
    changes = calculate_raster_change(
        result_2020,
        result_2021,
        categorical=False
    )
    
    print(f"\n温度变化:")
    for cell_token, change_data in changes.items():
        print(f"  单元格 {cell_token}:")
        print(f"    2020: {change_data['before']:.2f}°C")
        print(f"    2021: {change_data['after']:.2f}°C")
        print(f"    变化: {change_data['change_value']:+.2f}°C ({change_data['change_percent']:+.1f}%)")


####################################################################
# 示例 6-8: CDL 案例研究 - 使用通用栅格 API 处理特定领域数据
####################################################################

def example_6_cdl_basic_discretization():
    """示例 6: CDL 基础离散化 - 作物分布映射"""
    print("\n" + "="*70)
    print("示例 6: CDL 基础离散化 - 作物分布映射")
    print("="*70)
    print("演示: 使用通用栅格 API 处理 USDA CDL 数据")
    print("CDL (Cropland Data Layer) 是 USDA NASS 提供的年度栅格作物覆盖数据")
    print("30m 分辨率，254+ 种作物类型，可用于农业土地使用分析")
    
    # 创建样本 CDL 数据 (模拟艾奥瓦州玉米带)
    pixels = []
    
    # 玉米区域 (code 1) - 50 个像素
    for i in range(50):
        lat = 40.70 + (i % 10) * 0.001
        lon = -94.00 + (i // 10) * 0.001
        pixels.append(CDLPixel(lat=lat, lon=lon, crop_code=1, year=2021, confidence=95.0))
    
    # 大豆区域 (code 5) - 40 个像素
    for i in range(40):
        lat = 40.71 + (i % 8) * 0.001
        lon = -94.01 + (i // 8) * 0.001
        pixels.append(CDLPixel(lat=lat, lon=lon, crop_code=5, year=2021, confidence=92.0))
    
    # 冬小麦区域 (code 24) - 30 个像素
    for i in range(30):
        lat = 40.72 + (i % 6) * 0.001
        lon = -94.02 + (i // 6) * 0.001
        pixels.append(CDLPixel(lat=lat, lon=lon, crop_code=24, year=2021, confidence=90.0))
    
    # 使用通用栅格 API
    result = discretize_raster_categorical(
        pixels,
        level=12,
        name_mapping=CDL_CROP_CODES
    )
    
    print(f"\n✅ 离散化了 {len(result)} 个 DGGS 单元格")
    print(f"   总像素数: {len(pixels)}")
    
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  总像素: {data['total_pixels']}")
        print(f"  面积: {data['total_area_acres']:.2f} acres")
        print(f"  主导作物: {data['dominant_category']['name']} ({data['dominant_category']['percent']:.1f}%)")
        print(f"  作物多样性指数: {data['category_diversity']:.3f}")
        print(f"  作物分布:")
        for crop_name, crop_info in sorted(data['categories'].items(), key=lambda x: x[1]['percent'], reverse=True):
            print(f"    - {crop_name}: {crop_info['percent']:.1f}% ({crop_info['count']} pixels)")


def example_6b_cdl_crop_rotation():
    """示例 6b: CDL 作物轮作模式检测"""
    print("\n" + "="*70)
    print("示例 6b: CDL 作物轮作模式检测 - 多年分析")
    print("="*70)
    print("演示: 使用 CDL 多年数据识别农业轮作模式")
    print("轮作是可持续农业的关键 - 检测玉米-大豆-冬小麦轮作模式")
    
    # 2019 年数据
    pixels_2019 = [
        CDLPixel(lat=40.70, lon=-94.00, crop_code=1, year=2019, confidence=95.0),  # Corn
        CDLPixel(lat=40.70, lon=-94.01, crop_code=5, year=2019, confidence=92.0),  # Soybeans
        CDLPixel(lat=40.70, lon=-94.02, crop_code=24, year=2019, confidence=90.0),  # Winter wheat
    ]
    
    # 2020 年数据 (部分轮作)
    pixels_2020 = [
        CDLPixel(lat=40.70, lon=-94.00, crop_code=5, year=2020, confidence=92.0),  # Soybeans (rotated from corn)
        CDLPixel(lat=40.70, lon=-94.01, crop_code=24, year=2020, confidence=88.0),  # Winter wheat (rotated from soybean)
        CDLPixel(lat=40.70, lon=-94.02, crop_code=1, year=2020, confidence=93.0),  # Corn (rotated from wheat)
    ]
    
    # 2021 年数据 (继续轮作)
    pixels_2021 = [
        CDLPixel(lat=40.70, lon=-94.00, crop_code=24, year=2021, confidence=89.0),  # Winter wheat
        CDLPixel(lat=40.70, lon=-94.01, crop_code=1, year=2021, confidence=94.0),  # Corn
        CDLPixel(lat=40.70, lon=-94.02, crop_code=5, year=2021, confidence=91.0),  # Soybeans
    ]
    
    # 使用 CDL 轮作检测函数
    rotation_patterns = discretize_cdl_rotation_patterns(
        pixels_by_year={2019: pixels_2019, 2020: pixels_2020, 2021: pixels_2021},
        level=12
    )
    
    print(f"\n✅ 检测了 {len(rotation_patterns)} 个 DGGS 单元格的轮作模式")
    
    for cell_token, pattern_data in rotation_patterns.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  轮作序列: {' → '.join(pattern_data['rotation_sequence'])}")
        print(f"  规律性指数: {pattern_data['pattern_regularity']:.2f} (0-1)")
        print(f"  年份数: {pattern_data['num_years']}")
        print(f"  作物转换:")
        for from_crop, transitions in pattern_data['transitions'].items():
            for to_crop, count in transitions.items():
                print(f"    - {from_crop} → {to_crop}: {count} times")


def example_6c_cdl_agricultural_intensity():
    """示例 6c: CDL 农业强度分析"""
    print("\n" + "="*70)
    print("示例 6c: CDL 农业强度分析")
    print("="*70)
    print("演示: 根据作物类型和多样性评估农业强度")
    print("强度评分: 密集单一种植 (高强度) vs 多样化混合农业 (低强度)")
    
    # 密集单一种植区 (高强度)
    intensive_pixels = []
    for i in range(80):  # 80% 玉米
        lat = 40.70 + (i % 10) * 0.001
        lon = -94.00 + (i // 10) * 0.001
        intensive_pixels.append(CDLPixel(lat=lat, lon=lon, crop_code=1, year=2021))
    
    for i in range(20):  # 20% 其他
        lat = 40.705 + (i % 5) * 0.001
        lon = -94.005 + (i // 5) * 0.001
        intensive_pixels.append(CDLPixel(lat=lat, lon=lon, crop_code=5, year=2021))
    
    # 多样化农业区 (低强度)
    diverse_pixels = []
    crops = [1, 5, 24, 36, 42]  # Corn, Soybean, Wheat, Alfalfa, Beans
    for i, crop_code in enumerate(crops * 4):  # 均匀分布
        lat = 40.80 + (i % 10) * 0.001
        lon = -94.10 + (i // 10) * 0.001
        diverse_pixels.append(CDLPixel(lat=lat, lon=lon, crop_code=crop_code, year=2021))
    
    # 分析
    intensive_result = discretize_cdl_agricultural_intensity(intensive_pixels, level=12)
    diverse_result = discretize_cdl_agricultural_intensity(diverse_pixels, level=12)
    
    print("\n🌽 高强度农业区 (主要玉米单一种植):")
    for cell_token, intensity_data in intensive_result.items():
        print(f"  单元格 {cell_token}:")
        print(f"    强度等级: {intensity_data['intensity'].upper()}")
        print(f"    强度评分: {intensity_data['intensity_score']:.1f}/100")
        print(f"    农业面积占比: {intensity_data['ag_percent']:.1f}%")
        print(f"    是否为单一种植: {intensity_data['monoculture']}")
        print(f"    主导作物: {intensity_data['dominant_crop']}")
        print(f"    作物多样性: {intensity_data['crop_diversity']:.3f}")
    
    print("\n🌲 多样化农业区 (混合种植多种作物):")
    for cell_token, intensity_data in diverse_result.items():
        print(f"  单元格 {cell_token}:")
        print(f"    强度等级: {intensity_data['intensity'].upper()}")
        print(f"    强度评分: {intensity_data['intensity_score']:.1f}/100")
        print(f"    农业面积占比: {intensity_data['ag_percent']:.1f}%")
        print(f"    是否为单一种植: {intensity_data['monoculture']}")
        print(f"    主导作物: {intensity_data['dominant_crop']}")
        print(f"    作物多样性: {intensity_data['crop_diversity']:.3f}")


def example_7_custom_aggregation():
    """示例 7: 自定义聚合函数"""
    print("\n" + "="*70)
    print("示例 7: 自定义聚合函数 - 百分位数计算")
    print("="*70)
    
    pixels = [
        ContinuousPixel(lat=40.0, lon=-100.0, value=10.0),
        ContinuousPixel(lat=40.0, lon=-100.01, value=20.0),
        ContinuousPixel(lat=40.0, lon=-100.02, value=30.0),
        ContinuousPixel(lat=40.0, lon=-100.03, value=40.0),
    ]
    
    # 自定义聚合: 计算第 75 百分位数
    def percentile_75(values):
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * 0.75)
        return sorted_vals[idx]
    
    result = discretize_raster_continuous(
        pixels,
        level=12,
        aggregation_func='custom',
        custom_aggregator=percentile_75
    )
    
    print(f"\n✅ 使用自定义聚合函数")
    for cell_token, data in result.items():
        print(f"  单元格 {cell_token}:")
        print(f"    75th 百分位: {data['value']:.2f}")
        print(f"    平均值: {data['mean']:.2f}")
        print(f"    最大值: {data['max']:.2f}")


def example_8_cdl_complete_crop_distribution():
    """示例 8: CDL 完整作物分布分析"""
    print("\n" + "="*70)
    print("示例 8: CDL 完整作物分布分析")
    print("="*70)
    
    pixels = create_cdl_sample_data(year=2021)
    result = discretize_cdl_crop_distribution(pixels, level=12)
    
    print(f"\n✅ 作物分布分析 - {len(result)} 个单元格:\n")
    
    for cell_token, data in list(result.items())[:2]:
        print(f"  单元格 {cell_token}:")
        print(f"    - 总面积: {data['total_area_acres']:.2f} 英亩")
        print(f"    - 作物多样性: {data['crop_diversity']:.3f}")
        print(f"    - 主导作物: {data['dominant_crop']['name']} ({data['dominant_crop']['percent']:.1f}%)")
        print(f"    - 所有作物:")
        for crop_name, crop_data in sorted(data['crops'].items(), 
                                          key=lambda x: x[1]['percent'], 
                                          reverse=True):
            print(f"        • {crop_name}: {crop_data['percent']:.1f}% ({crop_data['count']} 像素, {crop_data['area_acres']:.3f} 英亩)")
        print()


def example_9_cdl_crop_categories():
    """示例 9: CDL 作物分类统计"""
    print("\n" + "="*70)
    print("示例 9: CDL 作物分类统计")
    print("="*70)
    
    pixels = create_cdl_sample_data(year=2021)
    result = discretize_cdl_crop_categories(pixels, level=12)
    
    print(f"\n✅ 作物分类统计 - {len(result)} 个单元格:\n")
    
    for cell_token, data in list(result.items())[:3]:
        print(f"  单元格 {cell_token}:")
        print(f"    - 农业用地: {data['agricultural_percent']:.1f}%")
        print(f"    - 主导类别: {data['dominant_category']['name']} ({data['dominant_category']['percent']:.1f}%)")
        print(f"    - 所有类别:")
        for category, cat_data in sorted(data['categories'].items(), 
                                        key=lambda x: x[1]['percent'], 
                                        reverse=True):
            print(f"        • {category}: {cat_data['percent']:.1f}% ({cat_data['count']} 像素)")
        print()


def example_10_cdl_temporal_analysis():
    """示例 10: CDL 时间序列作物分析"""
    print("\n" + "="*70)
    print("示例 10: CDL 时间序列作物分析")
    print("="*70)
    
    # 创建多年 CDL 数据
    pixels_by_year = {}
    for year in [2019, 2020, 2021]:
        pixels_by_year[year] = create_cdl_sample_data(year=year)
        # 模拟每年的变化
        if year == 2020:
            for p in pixels_by_year[year][:2]:
                if p.crop_code == 1:
                    p.crop_code = 5
                    p.category_name = "Soybeans"
    
    # 分析时间变化
    result = {}
    for year, pixels in pixels_by_year.items():
        year_result = discretize_cdl_crop_distribution(pixels, level=12)
        for cell_token, data in year_result.items():
            if cell_token not in result:
                result[cell_token] = {}
            result[cell_token][year] = data
    
    print(f"\n✅ 时间序列作物分析 - {len(result)} 个单元格，{len(pixels_by_year)} 年:\n")
    
    for cell_token, year_data in list(result.items())[:2]:
        print(f"  单元格 {cell_token}:")
        for year in sorted(year_data.keys()):
            data = year_data[year]
            print(f"    {year}: {data['dominant_crop']['name']} (多样性: {data['crop_diversity']:.3f})")
        print()


def example_11_cdl_rotation_patterns():
    """示例 11: CDL 作物轮作模式检测"""
    print("\n" + "="*70)
    print("示例 11: CDL 作物轮作模式检测")
    print("="*70)
    
    # 创建多年数据模拟轮作
    pixels_by_year = {}
    for year in range(2017, 2023):
        pixels_by_year[year] = create_cdl_sample_data(year=year)
        
        # 模拟轮作: 玉米 -> 大豆 -> 玉米
        cycle = (year - 2017) % 2
        for i, p in enumerate(pixels_by_year[year]):
            if p.crop_code == 1 and cycle == 1:
                p.crop_code = 5
                p.category_name = "Soybeans"
    
    # 检测轮作模式
    result = discretize_cdl_rotation_patterns(pixels_by_year, level=12, min_years=2)
    
    print(f"\n✅ 检测到轮作模式 - {len(result)} 个单元格:\n")
    
    for cell_token, data in list(result.items())[:3]:
        print(f"  单元格 {cell_token}:")
        print(f"    - 轮作序列 ({data['num_years']} 年): {' → '.join(data['rotation_sequence'])}")
        print(f"    - 模式规律性: {data['pattern_regularity']:.2f}")
        print(f"    - 作物转换:")
        for from_crop, transitions in data['transitions'].items():
            for to_crop, count in transitions.items():
                print(f"        • {from_crop} → {to_crop}: {count}次")
        print()


def example_12_cdl_knowledge_graph():
    """示例 12: CDL 知识图谱数据准备"""
    print("\n" + "="*70)
    print("示例 12: CDL 知识图谱数据准备")
    print("="*70)
    
    pixels = create_cdl_sample_data(year=2021)
    
    # 获取多个视角的 CDL 数据
    pixels_dist = discretize_raster_categorical(pixels, level=12)
    crop_dist = discretize_cdl_crop_distribution(pixels, level=12)
    categories = discretize_cdl_crop_categories(pixels, level=12)
    intensity = discretize_cdl_agricultural_intensity(pixels, level=12)
    
    print(f"\n📊 集成 CDL 知识用于知识图谱:\n")
    
    # 合并所有视角
    for cell_token in list(crop_dist.keys())[:1]:
        if cell_token not in categories or cell_token not in intensity:
            continue
        
        kg_entity = {
            'cell_token': cell_token,
            'spatial': {
                'level': 12,
                'total_pixels': pixels_dist[cell_token]['total_pixels'],
                'total_area_acres': crop_dist[cell_token]['total_area_acres']
            },
            'crops': {
                'dominant': crop_dist[cell_token]['dominant_crop']['name'],
                'diversity': crop_dist[cell_token]['crop_diversity'],
                'composition': {
                    crop: data['percent']
                    for crop, data in list(crop_dist[cell_token]['crops'].items())[:3]
                }
            },
            'land_use': {
                'dominant_category': categories[cell_token]['dominant_category']['name'],
                'agricultural_percent': categories[cell_token]['agricultural_percent'],
                'categories': {
                    cat: data['percent']
                    for cat, data in list(categories[cell_token]['categories'].items())[:3]
                }
            },
            'intensity': {
                'level': intensity[cell_token]['intensity'],
                'score': intensity[cell_token]['intensity_score'],
                'monoculture': intensity[cell_token]['monoculture']
            }
        }
        
        print(f"  📍 单元格 {cell_token}:")
        print(f"     空间: {kg_entity['spatial']['total_pixels']} 像素, {kg_entity['spatial']['total_area_acres']:.1f} 英亩")
        print(f"     作物: {kg_entity['crops']['dominant']} (多样性: {kg_entity['crops']['diversity']:.3f})")
        print(f"     组成: {kg_entity['crops']['composition']}")
        print(f"     土地利用: {kg_entity['land_use']['dominant_category']} ({kg_entity['land_use']['agricultural_percent']:.1f}% 农业)")
        print(f"     强度: {kg_entity['intensity']['level'].upper()} ({kg_entity['intensity']['score']:.1f}/100)")
        print(f"\n     知识图谱边:")
        print(f"       - 单元格 --has_dominant_crop--> {kg_entity['crops']['dominant']}")
        print(f"       - 单元格 --in_category--> {kg_entity['land_use']['dominant_category']}")
        print(f"       - 单元格 --intensity--> {kg_entity['intensity']['level']}")
        print(f"       - 单元格 --diversity--> {kg_entity['crops']['diversity']:.2f}")


if __name__ == "__main__":
    import sys
    
    print("\n🌍 通用栅格离散化示例与 CDL 案例研究")
    print("="*70)
    
    # 检查是否指定运行模式
    if len(sys.argv) > 1 and sys.argv[1] == '--cdl-only':
        # 仅运行 CDL 示例
        print("\n🌽 CDL 完整示例集")
        example_6_cdl_basic_discretization()
        example_8_cdl_complete_crop_distribution()
        example_9_cdl_crop_categories()
        example_6c_cdl_agricultural_intensity()
        example_10_cdl_temporal_analysis()
        example_6b_cdl_crop_rotation()
        example_11_cdl_rotation_patterns()
        example_12_cdl_knowledge_graph()
        
        print("\n" + "="*70)
        print("✅ 所有 CDL 示例完成!")
        print("="*70)
    else:
        # 运行所有示例
        # 通用栅格示例 (1-5)
        print("\n📚 第一部分: 通用栅格离散化示例")
        example_1_categorical_land_cover()
        example_2_continuous_temperature()
        example_3_continuous_elevation()
        example_4_temporal_series()
        example_5_temperature_change()
        
        # CDL 案例研究 (6-12)
        print("\n\n🌽 第二部分: CDL 作为通用栅格模块的案例研究")
        example_6_cdl_basic_discretization()
        example_8_cdl_complete_crop_distribution()
        example_9_cdl_crop_categories()
        example_6c_cdl_agricultural_intensity()
        example_10_cdl_temporal_analysis()
        example_6b_cdl_crop_rotation()
        example_11_cdl_rotation_patterns()
        example_12_cdl_knowledge_graph()
        
        # 其他示例 (7)
        print("\n\n🔧 第三部分: 高级功能")
        example_7_custom_aggregation()
        
        print("\n" + "="*70)
        print("✅ 所有示例完成!")
        print("="*70)
        print("\n💡 总结 - 通用栅格模块 (raster.py) 的功能:")
        print("  - 分类栅格: 土地覆盖, 作物类型, 土壤类型等")
        print("  - 连续栅格: 温度, 降水, 高程, NDVI 等")
        print("  - 时间序列: 多年数据, 变化检测, 趋势分析")
        print("  - 自定义聚合: 百分位数, 加权平均, 自定义函数等")
        print("\n🎯 架构设计:")
        print("  - 通用 API: discretize_raster_categorical/continuous/temporal")
        print("  - CDL 扩展: 基于通用 API 的专门分析函数")
        print("  - CDL 示例: 完整演示如何为特定数据类型实现高级分析")
        print("\n📖 运行方式:")
        print("  - python examples/raster_examples.py           # 运行所有示例")
        print("  - python examples/raster_examples.py --cdl-only # 仅运行 CDL 示例")
