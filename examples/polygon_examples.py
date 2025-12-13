"""
通用多边形离散化示例
Generic Polygon Discretization Examples

演示如何使用通用的 polygon.py 模块处理不同类型的矢量多边形数据：
1. 土地地块 (Land Parcels)
2. 行政单元 (Administrative Units)
3. 生态区 (Ecological Zones)
4. 流域单元 (Watersheds)
5. SSURGO 土壤数据 (作为多边形数据的案例研究)
"""

from pathlib import Path
import sys

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from DGGS import (
    PolygonFeature,
    PolygonComponent,
    discretize_polygon_features,
    discretize_polygon_attributes,
    discretize_polygon_categorical,
    discretize_polygon_hierarchical,
    SpatialEntity,
)
from DGGS.dggs import DGGSS2
from typing import Any, Dict, List, Optional, Sequence, Tuple
import statistics


####################################################################
# SSURGO 数据模型 - 基于通用多边形框架的扩展
####################################################################

class SSURGOMapUnit(PolygonFeature):
    """SSURGO 地图单元多边形类
    
    SSURGO (Soil Survey Geographic Database) 是 USDA NRCS 提供的详细土壤调查数据。
    这个类展示如何扩展通用的 PolygonFeature 来处理特定领域的数据。
    """
    
    def __init__(self, mukey: str, polygon_coords: Sequence[Tuple[float, float]], 
                 components: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            mukey: 地图单元唯一标识符 (Map Unit Key)
            polygon_coords: 多边形坐标列表 [(lat, lon), ...]
            components: 土壤组分列表，每个组分包含:
                - series_name: 土壤系列名称
                - percentage: 组分百分比
                - pH, sand_percent, clay_percent 等土壤属性
        """
        component_dicts = components or []
        
        # 将字典格式的组分转换为 PolygonComponent 对象
        polygon_components = []
        for comp_dict in component_dicts:
            comp = PolygonComponent(
                name=comp_dict.get('series_name', comp_dict.get('component_name', 'Unknown')),
                percentage=comp_dict.get('percentage', 0),
                attributes=comp_dict
            )
            polygon_components.append(comp)
        
        # 初始化父类
        super().__init__(
            feature_id=mukey,
            polygon_coords=polygon_coords,
            components=polygon_components,  # 使用 PolygonComponent 对象
            feature_type='ssurgo_map_unit'
        )
        
        # SSURGO 特定属性（保持向后兼容）
        self.mukey = mukey
        self.set_attribute('mukey', mukey)
        self.component_dicts = component_dicts  # 保存字典格式用于向后兼容
    
    def get_dominant_component(self) -> Optional[Dict[str, Any]]:
        """获取主导（占比最高）的土壤组分"""
        if not self.component_dicts:
            return None
        return max(self.component_dicts, key=lambda c: c.get('percentage', 0))
    
    def get_weighted_properties(self, property_name: str, 
                               aggregation: str = 'mean') -> Optional[float]:
        """计算组分加权的属性值
        
        Args:
            property_name: 属性名称 (如 'pH', 'sand_percent')
            aggregation: 聚合方式 ('mean', 'weighted_mean', 'max', 'min')
        
        Returns:
            聚合后的属性值
        """
        if not self.components:  # self.components 是 PolygonComponent 对象列表
            return None
        
        values = []
        weights = []
        
        for comp in self.components:
            val = comp.get_attribute(property_name)
            if val is not None:
                try:
                    values.append(float(val))
                    weights.append(comp.percentage)
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return None
        
        if aggregation == 'weighted_mean':
            total_weight = sum(weights)
            if total_weight == 0:
                return statistics.mean(values)
            return sum(v * w for v, w in zip(values, weights)) / total_weight
        elif aggregation == 'mean':
            return statistics.mean(values)
        elif aggregation == 'sum':
            return sum(values)
        elif aggregation == 'max':
            return max(values)
        elif aggregation == 'min':
            return min(values)
        elif aggregation == 'median':
            return statistics.median(values)
        
        return None


####################################################################
# SSURGO 分析函数 - 使用通用 API 实现的专用功能
####################################################################

def discretize_ssurgo_map_units(
    map_units: Sequence[SSURGOMapUnit],
    level: int = 12,
    method: str = 'centroid'
) -> Dict[str, Dict[str, Any]]:
    """将 SSURGO 地图单元离散化到 DGGS 单元格
    
    Args:
        map_units: SSURGOMapUnit 对象列表
        level: DGGS 层级 (10-14 推荐)
        method: 'centroid' (质心) 或 'coverage' (覆盖)
    
    Returns:
        字典 {cell_token: {'mukey': ..., 'dominant_component': ..., ...}}
    """
    # 使用通用多边形离散化
    result = discretize_polygon_features(map_units, level=level, method=method)
    
    # 添加 SSURGO 特定字段
    for cell_token, data in result.items():
        data['mukey'] = data.pop('feature_id')
        
        # 获取原始地图单元以获取主导组分
        mu = next((m for m in map_units if m.mukey == data['mukey']), None)
        if mu:
            data['dominant_component'] = mu.get_dominant_component()
    
    return result

####################################################################
# SSURGO 属性离散化函数
####################################################################

def discretize_ssurgo_soil_properties(
    map_units: Sequence[SSURGOMapUnit],
    properties: List[str],
    level: int = 12,
    aggregation_funcs: Optional[Dict[str, str]] = None,
    weight_by_component: bool = True
) -> Dict[str, Dict[str, Any]]:
    """离散化 SSURGO 土壤属性并进行统计聚合
    
    Args:
        map_units: SSURGOMapUnit 对象列表
        properties: 要聚合的属性列表 (如 ['pH', 'sand_percent'])
        level: DGGS 层级
        aggregation_funcs: 属性 -> 聚合函数的映射
        weight_by_component: 是否按组分百分比加权
    
    Returns:
        字典 {cell_token: {property: value, ...}}
    """
    result = discretize_polygon_attributes(
        map_units,
        attributes=properties,
        level=level,
        aggregation_funcs=aggregation_funcs,
        weight_by_component=weight_by_component
    )
    
    # 重命名字段以保持 SSURGO 术语
    for cell_token, data in result.items():
        if 'feature_id' in data:
            data['mukey'] = data.pop('feature_id')
        if 'num_components' in data:
            data['components_count'] = data.pop('num_components')
    
    return result

####################################################################
# SSURGO 作物适宜性离散化函数
####################################################################

def discretize_ssurgo_agricultural_suitability(
    map_units: Sequence[SSURGOMapUnit],
    crop: str = 'corn',
    level: int = 12
) -> Dict[str, Dict[str, Any]]:
    """计算作物农业适宜性评级
    
    基于关键土壤属性评估:
    - pH 范围
    - 排水等级
    - 质地
    - 深度
    
    Args:
        map_units: SSURGOMapUnit 对象列表
        crop: 作物类型 ('corn', 'wheat', 'soybean', 'alfalfa')
        level: DGGS 层级
    
    Returns:
        字典 {cell_token: {'suitability_class': ..., 'rating': 0-100, ...}}
    """
    # 不同作物的最佳土壤条件
    crop_requirements = {
        'corn': {'pH': (6.0, 7.5), 'drainage': 'well', 'texture': 'loam'},
        'wheat': {'pH': (6.0, 8.0), 'drainage': 'well', 'texture': 'loam'},
        'soybean': {'pH': (6.0, 7.5), 'drainage': 'well', 'texture': 'clay_loam'},
        'alfalfa': {'pH': (6.5, 8.0), 'drainage': 'well', 'texture': 'loam'},
    }
    
    reqs = crop_requirements.get(crop, crop_requirements['corn'])
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for mu in map_units:
        lat, lon = mu.centroid
        cell_token = grid.latlon_to_token(lat, lon, level)
        
        dominant = mu.get_dominant_component()
        if not dominant:
            continue
        
        # 计算适宜性评分 (0-100)
        score = 100
        
        # pH 因子
        ph = dominant.get('pH')
        if ph:
            ph_min, ph_max = reqs['pH']
            if ph < ph_min or ph > ph_max:
                score *= 0.7
            elif abs(ph - ((ph_min + ph_max) / 2)) > 0.5:
                score *= 0.85
        
        # 排水因子
        drainage = dominant.get('drainage_class')
        if drainage != reqs['drainage']:
            score *= 0.75
        
        # 确定适宜性等级
        if score >= 80:
            suitability = 'Highly Suitable'
        elif score >= 60:
            suitability = 'Suitable'
        elif score >= 40:
            suitability = 'Marginally Suitable'
        else:
            suitability = 'Not Suitable'
        
        result[cell_token] = {
            'mukey': mu.mukey,
            'crop': crop,
            'suitability_class': suitability,
            'score': round(score, 1),
            'dominant_series': dominant.get('series_name', 'Unknown'),
            'dominant_component_pct': dominant.get('percentage', 0)
        }
    
    return result

####################################################################
# SSURGO 水文土壤组离散化函数
####################################################################

def discretize_ssurgo_hydrologic_group(
    map_units: Sequence[SSURGOMapUnit],
    level: int = 12
) -> Dict[str, Dict[str, Any]]:
    """离散化水文土壤组 (HSG) 用于径流/渗透分析
    
    USDA 水文土壤组:
    - A: 低径流，高渗透 (沙质)
    - B: 低-中径流 (壤质砂土到壤土)
    - C: 中-高径流 (砂质粘土到粘土)
    - D: 高径流，低渗透 (粘土)
    
    Args:
        map_units: SSURGOMapUnit 对象列表
        level: DGGS 层级
    
    Returns:
        字典 {cell_token: {'hsg': 'A'|'B'|'C'|'D', 'infiltration': ..., ...}}
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    # 各水文组的渗透率 (英寸/小时，近似值)
    infiltration_rates = {
        'A': 0.8,
        'B': 0.25,
        'C': 0.1,
        'D': 0.05
    }
    
    for mu in map_units:
        lat, lon = mu.centroid
        cell_token = grid.latlon_to_token(lat, lon, level)
        
        dominant = mu.get_dominant_component()
        if not dominant:
            continue
        
        hsg = dominant.get('hydro_group', 'B')
        primary_hsg = hsg.split('/')[0]  # 处理双重分类 (如 A/D)
        
        result[cell_token] = {
            'mukey': mu.mukey,
            'hydro_group': hsg,
            'primary_hsg': primary_hsg,
            'infiltration_in_hr': infiltration_rates.get(primary_hsg, 0.25),
            'dominant_component_pct': dominant.get('percentage', 0),
            'num_components': len(mu.component_dicts)
        }
    
    return result


####################################################################
# SSURGO 工具函数
####################################################################

def create_ssurgo_sample_data() -> List[SSURGOMapUnit]:
    """创建 SSURGO 样例数据用于演示"""
    
    map_units = [
        SSURGOMapUnit(
            mukey='123001',
            polygon_coords=[(40.70, -74.00), (40.71, -74.00), (40.71, -74.01), (40.70, -74.01)],
            components=[
                {
                    'series_name': 'Inwood',
                    'percentage': 70,
                    'pH': 6.8,
                    'sand_percent': 25,
                    'clay_percent': 35,
                    'silt_percent': 40,
                    'bulk_density': 1.4,
                    'drainage_class': 'well',
                    'hydro_group': 'B',
                    'ksat': 0.5
                },
                {
                    'series_name': 'Yonkers',
                    'percentage': 30,
                    'pH': 7.1,
                    'sand_percent': 35,
                    'clay_percent': 25,
                    'silt_percent': 40,
                    'bulk_density': 1.3,
                    'drainage_class': 'well',
                    'hydro_group': 'A/B',
                    'ksat': 0.8
                }
            ]
        ),
        SSURGOMapUnit(
            mukey='123002',
            polygon_coords=[(40.71, -74.00), (40.72, -74.00), (40.72, -74.01), (40.71, -74.01)],
            components=[
                {
                    'series_name': 'Central Park',
                    'percentage': 100,
                    'pH': 6.2,
                    'sand_percent': 60,
                    'clay_percent': 10,
                    'silt_percent': 30,
                    'bulk_density': 1.2,
                    'drainage_class': 'excessive',
                    'hydro_group': 'A',
                    'ksat': 2.0
                }
            ]
        ),
        SSURGOMapUnit(
            mukey='123003',
            polygon_coords=[(40.72, -74.00), (40.73, -74.00), (40.73, -74.01), (40.72, -74.01)],
            components=[
                {
                    'series_name': 'Clarion',
                    'percentage': 60,
                    'pH': 6.5,
                    'sand_percent': 35,
                    'clay_percent': 25,
                    'silt_percent': 40,
                    'bulk_density': 1.35,
                    'drainage_class': 'well',
                    'hydro_group': 'B',
                    'ksat': 0.6
                },
                {
                    'series_name': 'Webster',
                    'percentage': 40,
                    'pH': 7.0,
                    'sand_percent': 25,
                    'clay_percent': 35,
                    'silt_percent': 40,
                    'bulk_density': 1.45,
                    'drainage_class': 'poor',
                    'hydro_group': 'C/D',
                    'ksat': 0.15
                }
            ]
        ),
    ]
    
    return map_units


def parse_ssurgo_csv(csv_file_path: str) -> List[SSURGOMapUnit]:
    """从 CSV 文件解析 SSURGO 数据
    
    CSV 格式应包含: mukey,lat,lon,component_name,percentage,pH,sand_percent,...
    
    Args:
        csv_file_path: CSV 文件路径
    
    Returns:
        SSURGOMapUnit 对象列表
    """
    import csv
    
    map_units_dict: Dict[str, Dict] = {}
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mukey = row.get('mukey')
            if not mukey:
                continue
            
            if mukey not in map_units_dict:
                map_units_dict[mukey] = {
                    'lat': float(row.get('lat', 0)),
                    'lon': float(row.get('lon', 0)),
                    'components': []
                }
            
            # 从 CSV 行构建组分
            component = {
                'series_name': row.get('component_name', ''),
                'percentage': float(row.get('percentage', 0)),
            }
            
            # 添加数值属性
            for key in row:
                if key not in ['mukey', 'lat', 'lon', 'component_name', 'percentage']:
                    try:
                        component[key] = float(row[key])
                    except (ValueError, TypeError):
                        component[key] = row[key]
            
            map_units_dict[mukey]['components'].append(component)
    
    # 转换为 SSURGOMapUnit 对象
    map_units = []
    for mukey, data in map_units_dict.items():
        lat, lon = data['lat'], data['lon']
        # 创建简单的质心多边形用于演示
        polygon = [(lat, lon), (lat + 0.01, lon), (lat + 0.01, lon + 0.01), 
                   (lat, lon + 0.01)]
        
        mu = SSURGOMapUnit(mukey, polygon, data['components'])
        map_units.append(mu)
    
    return map_units


####################################################################
# SSURGO 数据转知识图谱
####################################################################

def discretized_ssurgo_to_triplets(cell_token: str, ssurgo_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """将离散化的 SSURGO 土壤数据转换为 RDF 三元组
    
    此函数将 discretize_ssurgo_soil_properties() 输出转换为知识图谱三元组格式，
    便于整合到知识图谱系统中。
    
    Args:
        cell_token: DGGS 单元格 token (如 '89c25a3')
        ssurgo_data: discretize_ssurgo_soil_properties() 返回的字典，包含:
            - weighted_ph: 加权平均 pH 值
            - weighted_ksat: 加权饱和导水率
            - weighted_clay_pct: 加权黏土百分比
            - weighted_sand_pct: 加权砂土百分比
            - dominant_texture: 主导土壤质地
            - map_unit_count: 图斑单元数量
            - component_count: 土壤组分数量
    
    Returns:
        RDF 三元组列表，格式为 (subject, predicate, object) 元组
        
    Example:
        >>> data = discretize_ssurgo_soil_properties([map_unit1, map_unit2], level=12)
        >>> triplets = discretized_ssurgo_to_triplets('89c25a3', data['89c25a3'])
        >>> print(triplets[:3])
        [
            ('soil_89c25a3', 'rdf:type', 'SoilCell'),
            ('soil_89c25a3', 'attr:dggs_level', '12'),
            ('soil_89c25a3', 'soil_property', 'weighted:ph=6.45')
        ]
    """
    triplets = []
    
    # 1. 创建土壤单元格实体
    entity = SpatialEntity(
        f'soil_{cell_token}',
        'SoilCell',
        {
            'dggs_level': 12,
            'source': 'SSURGO'
        }
    )
    triplets.extend(entity.to_triplets())
    
    # 2. 添加土壤属性三元组（聚合后的数值）
    for prop, value in ssurgo_data.items():
        if prop not in ['map_unit_count', 'component_count'] and value is not None:
            # 规范化属性名（将下划线转为冒号，符合 RDF 命名空间约定）
            prop_clean = prop.replace('_', ':')
            
            # 格式化数值
            if isinstance(value, (int, float)):
                value_str = str(round(float(value), 2))
            else:
                value_str = str(value)
            
            triplets.append((
                f'soil_{cell_token}',
                f'soil_property',
                f'{prop_clean}={value_str}'
            ))
    
    return triplets


####################################################################
# 通用多边形示例 (1-5)
####################################################################

# 示例 1: 土地地块 (Land Parcels)
def example_1_land_parcels():
    """示例 1: 土地地块离散化"""
    print("\n" + "="*70)
    print("示例 1: 土地地块 (Land Parcels)")
    print("="*70)
    
    # 创建土地地块（多用途）
    parcels = [
        PolygonFeature(
            feature_id='parcel_001',
            polygon_coords=[(40.0, -100.0), (40.01, -100.0), (40.01, -100.01), (40.0, -100.01)],
            components=[
                PolygonComponent('residential', 70.0, {'building_count': 5, 'property_value': 500000}),
                PolygonComponent('commercial', 30.0, {'building_count': 1, 'property_value': 300000}),
            ],
            feature_type='land_parcel',
            attributes={'owner': 'City', 'zone': 'mixed_use'}
        ),
        PolygonFeature(
            feature_id='parcel_002',
            polygon_coords=[(40.01, -100.0), (40.02, -100.0), (40.02, -100.01), (40.01, -100.01)],
            components=[
                PolygonComponent('agricultural', 100.0, {'crop_type': 'corn', 'acres': 50}),
            ],
            feature_type='land_parcel',
            attributes={'owner': 'Farmer', 'zone': 'agricultural'}
        ),
    ]
    
    # 离散化
    result = discretize_polygon_features(parcels, level=12, method='centroid')
    
    print(f"\n✅ 离散化了 {len(result)} 个 DGGS 单元格")
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  地块 ID: {data['feature_id']}")
        print(f"  主导用途: {data['dominant_component']}")
        print(f"  用途多样性: {data['component_diversity']:.2f}")
        print(f"  分区: {data.get('polygon_zone', 'N/A')}")

# 示例 2: 行政单元 (Administrative Units)
def example_2_administrative_units():
    """示例 2: 行政单元（人口统计）"""
    print("\n" + "="*70)
    print("示例 2: 行政单元 - 人口统计")
    print("="*70)
    
    # 创建人口普查区
    census_tracts = [
        PolygonFeature(
            feature_id='tract_001',
            polygon_coords=[(40.0, -100.0), (40.05, -100.0), (40.05, -100.05), (40.0, -100.05)],
            components=[
                PolygonComponent('income_low', 40.0, {'median_income': 35000, 'population': 2000}),
                PolygonComponent('income_medium', 45.0, {'median_income': 55000, 'population': 2250}),
                PolygonComponent('income_high', 15.0, {'median_income': 95000, 'population': 750}),
            ],
            feature_type='census_tract',
            attributes={'total_population': 5000, 'area_sqmi': 2.5}
        ),
    ]
    
    # 离散化属性
    result = discretize_polygon_attributes(
        census_tracts,
        attributes=['median_income', 'population'],
        level=11,
        aggregation_funcs={'median_income': 'weighted_mean', 'population': 'sum'}
    )
    
    print(f"\n✅ 离散化人口统计数据")
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  加权平均收入: ${data['median_income_weighted_mean']:,.0f}")
        print(f"  总人口: {data['population_sum']:,.0f}")

# 示例 3: 生态区 (Ecological Zones)
def example_3_ecological_zones():
    """示例 3: 生态区（物种组成）"""
    print("\n" + "="*70)
    print("示例 3: 生态区 - 物种多样性")
    print("="*70)
    
    # 创建生态区
    habitats = [
        PolygonFeature(
            feature_id='habitat_001',
            polygon_coords=[(40.0, -105.0), (40.02, -105.0), (40.02, -105.02), (40.0, -105.02)],
            components=[
                PolygonComponent('oak', 45.0, {'tree_count': 150, 'avg_height_m': 12}),
                PolygonComponent('pine', 35.0, {'tree_count': 200, 'avg_height_m': 18}),
                PolygonComponent('aspen', 20.0, {'tree_count': 100, 'avg_height_m': 10}),
            ],
            feature_type='forest_habitat',
            attributes={'protected_status': 'yes', 'fire_risk': 'moderate'}
        ),
    ]
    
    # 分类离散化
    result = discretize_polygon_categorical(
        habitats,
        category_attribute='tree_count',
        level=12,
        method='diversity'
    )
    
    print(f"\n✅ 生态区离散化完成")
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  生境 ID: {data['feature_id']}")
        print(f"  物种多样性指数: {data.get('diversity', 0):.2f}")
        print(f"  物种数: {data.get('num_categories', 0)}")

# 示例 4: 流域单元 (Watersheds)
def example_4_watersheds():
    """示例 4: 流域单元（水文参数）"""
    print("\n" + "="*70)
    print("示例 4: 流域单元 - 水文特征")
    print("="*70)
    
    # 创建流域
    watersheds = [
        PolygonFeature(
            feature_id='ws_001',
            polygon_coords=[(40.0, -100.0), (40.1, -100.0), (40.1, -100.1), (40.0, -100.1)],
            components=[
                PolygonComponent('upstream', 30.0, {
                    'elevation_m': 1500,
                    'slope_pct': 5.0,
                    'runoff_coef': 0.3
                }),
                PolygonComponent('midstream', 50.0, {
                    'elevation_m': 1200,
                    'slope_pct': 2.0,
                    'runoff_coef': 0.4
                }),
                PolygonComponent('downstream', 20.0, {
                    'elevation_m': 900,
                    'slope_pct': 1.0,
                    'runoff_coef': 0.5
                }),
            ],
            feature_type='watershed',
            attributes={'stream_order': 3, 'drainage_area_sqkm': 150}
        ),
    ]
    
    # 离散化水文属性
    result = discretize_polygon_attributes(
        watersheds,
        attributes=['elevation_m', 'slope_pct', 'runoff_coef'],
        level=11,
        aggregation_funcs={
            'elevation_m': 'weighted_mean',
            'slope_pct': 'weighted_mean',
            'runoff_coef': 'weighted_mean'
        }
    )
    
    print(f"\n✅ 流域离散化完成")
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  平均高程: {data['elevation_m_weighted_mean']:.1f} m")
        print(f"  平均坡度: {data['slope_pct_weighted_mean']:.2f} %")
        print(f"  径流系数: {data['runoff_coef_weighted_mean']:.3f}")

# 示例 5: 层次化区域 (Hierarchical Zones)
def example_5_hierarchical_zones():
    """示例 5: 层次化区域（建筑物楼层）"""
    print("\n" + "="*70)
    print("示例 5: 层次化区域 - 建筑物楼层")
    print("="*70)
    
    # 创建多层建筑
    buildings = [
        PolygonFeature(
            feature_id='building_001',
            polygon_coords=[(40.0, -100.0), (40.001, -100.0), (40.001, -100.001), (40.0, -100.001)],
            components=[
                PolygonComponent('floor_1', 33.3, {
                    'use_type': 'retail',
                    'area_sqm': 500,
                    'occupancy': 100
                }),
                PolygonComponent('floor_2', 33.3, {
                    'use_type': 'office',
                    'area_sqm': 500,
                    'occupancy': 50
                }),
                PolygonComponent('floor_3', 33.3, {
                    'use_type': 'office',
                    'area_sqm': 500,
                    'occupancy': 50
                }),
            ],
            feature_type='building',
            attributes={'height_m': 12, 'year_built': 2010}
        ),
    ]
    
    # 层次化离散化
    hierarchy_levels = {
        'ground_floor': (1, 1),
        'upper_floors': (2, 10)
    }
    
    attributes_per_level = {
        'ground_floor': ['use_type', 'occupancy'],
        'upper_floors': ['use_type', 'occupancy']
    }
    
    result = discretize_polygon_hierarchical(
        buildings,
        hierarchy_levels=hierarchy_levels,
        attributes_per_level=attributes_per_level,
        level=13
    )
    
    print(f"\n✅ 层次化离散化完成")
    for cell_token, levels in result.items():
        print(f"\n单元格 {cell_token}:")
        for level_name, data in levels.items():
            print(f"  {level_name}: {data}")

####################################################################
# SSURGO 示例 (6-9)
####################################################################

# 示例 6: SSURGO 基础离散化
def example_6_ssurgo_basic_discretization():
    """示例 6: SSURGO 基础离散化"""
    print("\n" + "="*70)
    print("示例 6: SSURGO 土壤地图单元离散化")
    print("="*70)
    print("演示: 使用通用多边形 API 处理 USDA SSURGO 土壤数据")
    print("SSURGO 提供详细的土壤调查信息，包括多个土壤组分及其属性\n")
    
    # 创建样例 SSURGO 数据
    map_units = create_ssurgo_sample_data()
    
    # 方式 1: 使用 SSURGO 专用函数
    result_ssurgo = discretize_ssurgo_map_units(map_units, level=12, method='centroid')
    
    print(f"✅ 离散化了 {len(map_units)} 个地图单元到 {len(result_ssurgo)} 个 DGGS 单元格\n")
    
    for cell_token, data in result_ssurgo.items():
        print(f"单元格 {cell_token}:")
        print(f"  MUKEY: {data['mukey']}")
        print(f"  组分数: {data['num_components']}")
        dom = data['dominant_component']
        print(f"  主导组分: {dom['series_name']} ({dom['percentage']}%)")
        print(f"    - pH: {dom.get('pH', 'N/A')}")
        print(f"    - 砂含量: {dom.get('sand_percent', 'N/A')}%")
        print(f"    - 粘土含量: {dom.get('clay_percent', 'N/A')}%")
        print()

# 示例 7: SSURGO 土壤属性聚合
def example_7_ssurgo_soil_properties():
    """示例 7: SSURGO 土壤属性聚合"""
    print("\n" + "="*70)
    print("示例 7: SSURGO 土壤属性聚合")
    print("="*70)
    print("演示: 按组分百分比加权聚合土壤属性\n")
    
    map_units = create_ssurgo_sample_data()
    
    # 聚合关键土壤属性
    result = discretize_ssurgo_soil_properties(
        map_units,
        properties=['pH', 'sand_percent', 'clay_percent', 'bulk_density'],
        level=12,
        aggregation_funcs={
            'pH': 'weighted_mean',
            'sand_percent': 'weighted_mean',
            'clay_percent': 'weighted_mean',
            'bulk_density': 'weighted_mean'
        },
        weight_by_component=True
    )
    
    print(f"✅ 聚合了 {len(result)} 个单元格的土壤属性:\n")
    
    for cell_token, data in result.items():
        print(f"单元格 {cell_token}:")
        print(f"  MUKEY: {data['mukey']}")
        print(f"  加权平均 pH: {data.get('pH_weighted_mean', 'N/A'):.2f}")
        print(f"  加权平均砂含量: {data.get('sand_percent_weighted_mean', 'N/A'):.1f}%")
        print(f"  加权平均粘土含量: {data.get('clay_percent_weighted_mean', 'N/A'):.1f}%")
        print(f"  加权平均容重: {data.get('bulk_density_weighted_mean', 'N/A'):.2f} g/cm³")
        print()

# 示例 8: SSURGO 农业适宜性评估
def example_8_ssurgo_agricultural_suitability():
    """示例 8: SSURGO 农业适宜性评估"""
    print("\n" + "="*70)
    print("示例 8: SSURGO 农业适宜性评估")
    print("="*70)
    print("演示: 基于土壤属性评估作物适宜性\n")
    
    map_units = create_ssurgo_sample_data()
    
    # 评估玉米种植适宜性
    corn_suit = discretize_ssurgo_agricultural_suitability(map_units, crop='corn', level=12)
    
    print("🌽 玉米种植适宜性:\n")
    for cell_token, data in corn_suit.items():
        print(f"单元格 {cell_token}:")
        print(f"  土壤系列: {data['dominant_series']}")
        print(f"  适宜性等级: {data['suitability_class']}")
        print(f"  适宜性评分: {data['score']}/100")
        print()
    
    # 评估大豆种植适宜性
    soy_suit = discretize_ssurgo_agricultural_suitability(map_units, crop='soybean', level=12)
    
    print("🫛 大豆种植适宜性:\n")
    for cell_token, data in soy_suit.items():
        print(f"单元格 {cell_token}:")
        print(f"  土壤系列: {data['dominant_series']}")
        print(f"  适宜性等级: {data['suitability_class']}")
        print(f"  适宜性评分: {data['score']}/100")
        print()

# 示例 9: SSURGO 水文土壤组分析
def example_9_ssurgo_hydrologic_group():
    """示例 9: SSURGO 水文土壤组分析"""
    print("\n" + "="*70)
    print("示例 9: SSURGO 水文土壤组分析")
    print("="*70)
    print("演示: 分析土壤水文特性用于径流/渗透建模\n")
    
    map_units = create_ssurgo_sample_data()
    
    result = discretize_ssurgo_hydrologic_group(map_units, level=12)
    
    print(f"✅ 水文土壤组分析 - {len(result)} 个单元格:\n")
    
    for cell_token, data in result.items():
        print(f"单元格 {cell_token}:")
        print(f"  水文组: {data['hydro_group']}")
        print(f"  主要 HSG: {data['primary_hsg']}")
        print(f"  渗透率: {data['infiltration_in_hr']:.2f} 英寸/小时")
        
        # 解释水文组含义
        hsg_meanings = {
            'A': '低径流，高渗透 (沙质土壤)',
            'B': '低-中径流 (壤土)',
            'C': '中-高径流 (粘壤土)',
            'D': '高径流，低渗透 (粘土)'
        }
        meaning = hsg_meanings.get(data['primary_hsg'], '未知')
        print(f"  含义: {meaning}")
        print()

# 示例 10: 通用 API vs SSURGO 专用函数对比
def example_10_ssurgo_generic_vs_specific():
    """示例 10: 通用 API vs SSURGO 专用函数对比"""
    print("\n" + "="*70)
    print("示例 10: 通用 API vs SSURGO 专用函数对比")
    print("="*70)
    print("演示: SSURGOMapUnit 同时支持通用和专用接口\n")
    
    map_units = create_ssurgo_sample_data()[:1]  # 只用第一个
    
    # 方式 1: 使用通用多边形 API
    print("方式 1: 使用通用多边形 API")
    print("-" * 50)
    result_generic = discretize_polygon_features(map_units, level=12, method='centroid')
    
    for cell_token, data in result_generic.items():
        print(f"单元格 {cell_token}:")
        print(f"  特征 ID: {data['feature_id']}")
        print(f"  特征类型: {data['feature_type']}")
        print(f"  主导组分: {data['dominant_component']}")
        print(f"  组分多样性: {data['component_diversity']:.3f}")
    
    print()
    
    # 方式 2: 使用 SSURGO 专用函数
    print("方式 2: 使用 SSURGO 专用函数")
    print("-" * 50)
    result_ssurgo = discretize_ssurgo_map_units(map_units, level=12, method='centroid')
    
    for cell_token, data in result_ssurgo.items():
        print(f"单元格 {cell_token}:")
        print(f"  MUKEY: {data['mukey']}")
        dom = data['dominant_component']
        print(f"  主导土壤系列: {dom['series_name']} ({dom['percentage']}%)")
        print(f"  pH: {dom['pH']}")
        print(f"  排水等级: {dom['drainage_class']}")
    
    print("\n💡 两种方式都可以使用！")
    print("   - 通用 API: 适合与其他多边形数据一起处理")
    print("   - 专用函数: 提供 SSURGO 特定的字段名和功能")

# 示例 6 (扩展): SSURGO 使用通用 API
def example_6_ssurgo_using_generic_api():
    """示例 6: SSURGO 土壤数据 - 使用通用 API"""
    print("\n" + "="*70)
    print("示例 6: SSURGO 土壤 - 使用通用多边形 API")
    print("="*70)
    
    # SSURGOMapUnit 现在继承自 PolygonFeature
    map_units = [
        SSURGOMapUnit(
            mukey='123001',
            polygon_coords=[(40.0, -95.0), (40.01, -95.0), (40.01, -95.01), (40.0, -95.01)],
            components=[
                {
                    'series_name': 'Clarion',
                    'percentage': 70,
                    'pH': 6.5,
                    'sand_percent': 35,
                    'clay_percent': 25
                },
                {
                    'series_name': 'Webster',
                    'percentage': 30,
                    'pH': 7.0,
                    'sand_percent': 25,
                    'clay_percent': 35
                }
            ]
        ),
    ]
    
    # 方式 1: 使用 SSURGO 专用函数
    from DGGS import discretize_ssurgo_map_units
    result_ssurgo = discretize_ssurgo_map_units(map_units, level=12)
    
    print("\n方式 1: SSURGO 专用函数")
    for cell_token, data in result_ssurgo.items():
        print(f"  单元格 {cell_token}:")
        print(f"    MUKEY: {data['mukey']}")
        print(f"    主导组分: {data['dominant_component']['series_name']}")
    
    # 方式 2: 使用通用多边形函数
    result_generic = discretize_polygon_features(map_units, level=12)
    
    print("\n方式 2: 通用多边形函数")
    for cell_token, data in result_generic.items():
        print(f"  单元格 {cell_token}:")
        print(f"    特征 ID: {data['feature_id']}")
        print(f"    主导组分: {data['dominant_component']}")
        print(f"    组分多样性: {data['component_diversity']:.2f}")

# 示例 11: 多个多边形在同一单元格中的聚合
def example_11_multi_polygon_aggregation():
    """示例 11: 多个多边形在同一单元格中的聚合"""
    print("\n" + "="*70)
    print("示例 11: 多多边形聚合")
    print("="*70)
    
    # 创建多个小地块
    small_parcels = [
        PolygonFeature(
            f'parcel_{i:03d}',
            [(40.0 + i*0.001, -100.0), (40.0 + i*0.001 + 0.001, -100.0), 
             (40.0 + i*0.001 + 0.001, -100.001), (40.0 + i*0.001, -100.001)],
            [PolygonComponent('residential', 100.0, {'value': 100000 + i*10000})],
            feature_type='small_parcel'
        )
        for i in range(5)
    ]
    
    # 使用 coverage 方法可能会有多个地块在同一单元格
    result = discretize_polygon_features(small_parcels, level=12, method='centroid')
    
    print(f"\n✅ 创建了 {len(small_parcels)} 个地块")
    print(f"✅ 离散化为 {len(result)} 个单元格")
    
    for cell_token, data in result.items():
        print(f"\n单元格 {cell_token}:")
        print(f"  特征 ID: {data['feature_id']}")


if __name__ == "__main__":
    import sys
    
    print("\n🌍 通用多边形离散化示例与 SSURGO 案例研究")
    print("="*70)
    
    # 检查是否指定运行模式
    if len(sys.argv) > 1 and sys.argv[1] == '--ssurgo-only':
        # 仅运行 SSURGO 示例
        print("\n🌱 SSURGO 完整示例集")
        example_6_ssurgo_basic_discretization()
        example_7_ssurgo_soil_properties()
        example_8_ssurgo_agricultural_suitability()
        example_9_ssurgo_hydrologic_group()
        example_10_ssurgo_generic_vs_specific()
        
        print("\n" + "="*70)
        print("✅ 所有 SSURGO 示例完成!")
        print("="*70)
    else:
        # 运行所有示例
        print("\n📚 第一部分: 通用多边形离散化示例")
        example_1_land_parcels()
        example_2_administrative_units()
        example_3_ecological_zones()
        example_4_watersheds()
        example_5_hierarchical_zones()
        
        print("\n\n🌱 第二部分: SSURGO 作为通用多边形模块的案例研究")
        example_6_ssurgo_basic_discretization()
        example_7_ssurgo_soil_properties()
        example_8_ssurgo_agricultural_suitability()
        example_9_ssurgo_hydrologic_group()
        example_10_ssurgo_generic_vs_specific()
        
        print("\n\n🔧 第三部分: 其他示例")
        example_11_multi_polygon_aggregation()
        
        print("\n" + "="*70)
        print("✅ 所有示例完成!")
        print("="*70)
        print("\n💡 总结 - 通用多边形模块 (polygon.py) 的功能:")
        print("  - 土地地块: 分区、用途、产权")
        print("  - 行政单元: 人口、收入、统计")
        print("  - 生态区: 物种、栖息地、保护")
        print("  - 流域: 水文、地形、排水")
        print("  - 土壤: SSURGO、STATSGO 等")
        print("  - 任何多边形矢量数据!")
        print("\n🎯 架构设计:")
        print("  - 通用 API: discretize_polygon_features/attributes/categorical/hierarchical")
        print("  - SSURGO 扩展: 基于通用 API 的专门分析函数")
        print("  - SSURGO 示例: 完整演示如何为特定数据类型实现高级分析")
        print("\n📖 运行方式:")
        print("  - python examples/polygon_examples.py              # 运行所有示例")
        print("  - python examples/polygon_examples.py --ssurgo-only # 仅运行 SSURGO 示例")
