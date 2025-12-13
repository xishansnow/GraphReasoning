"""
DGGS Polygon Discretization Module

Generic discretization framework for polygon/vector data with attributes.
Supports any polygon-based spatial data with nested components/attributes.

Key concepts:
- Polygon: Geographic boundary defined by coordinate sequences
- Components: Nested features within polygons (with percentages/weights)
- Attributes: Properties associated with components or polygons
- Aggregation: Statistical combination of attributes across components

Supported data types:
- Soil surveys (SSURGO, STATSGO)
- Land parcels (cadastral, property boundaries)
- Administrative units (census, political boundaries)
- Ecological zones (ecoregions, habitats)
- Hydrological units (watersheds, aquifers)

Architecture:
    PolygonFeature (base class)
        ├── Component (nested features with weights)
        └── Discretization methods (centroid, coverage, weighted)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
from s2sphere import CellId
from .dggs import DGGSS2
from .spatial_utils import discretize_aggregate, discretize_weighted_aggregate, discretize_polygon_strict
import statistics


####################################################################
# Generic Polygon Data Models
####################################################################

class PolygonComponent:
    """Represents a component/subfeature within a polygon.
    
    Components are nested features within a polygon, each with:
    - Percentage/weight (relative abundance)
    - Attributes (properties/measurements)
    - Optional sub-components (hierarchical)
    
    Examples:
    - Soil components in a map unit
    - Land use types in a parcel
    - Species in a habitat
    - Income brackets in a census tract
    """
    
    def __init__(self, 
                 name: str,
                 percentage: float = 100.0,
                 attributes: Optional[Dict[str, Any]] = None,
                 sub_components: Optional[List['PolygonComponent']] = None):
        """
        Args:
            name: Component identifier/name
            percentage: Relative abundance (0-100)
            attributes: Dictionary of component properties
            sub_components: Nested components (for hierarchical data)
        """
        self.name = name
        self.percentage = percentage
        self.attributes = attributes or {}
        self.sub_components = sub_components or []
    
    def get_attribute(self, attr_name: str, default: Any = None) -> Any:
        """Get attribute value with fallback."""
        return self.attributes.get(attr_name, default)
    
    def set_attribute(self, attr_name: str, value: Any) -> None:
        """Set attribute value."""
        self.attributes[attr_name] = value


class PolygonFeature:
    """Generic polygon feature with components and attributes.
    
    Represents any polygon-based spatial feature that can contain:
    - Boundary coordinates
    - Multiple components (with percentages)
    - Attributes at polygon or component level
    - Hierarchical sub-components
    
    This is the base class for domain-specific implementations like
    SSURGO map units, land parcels, census tracts, etc.
    """
    
    def __init__(self, 
                 feature_id: str,
                 polygon_coords: Sequence[Tuple[float, float]],
                 components: Optional[List[PolygonComponent]] = None,
                 attributes: Optional[Dict[str, Any]] = None,
                 feature_type: Optional[str] = None):
        """
        Args:
            feature_id: Unique identifier for this polygon
            polygon_coords: List of (lat, lon) tuples forming polygon boundary
            components: List of PolygonComponent objects
            attributes: Polygon-level attributes (not component-specific)
            feature_type: Type descriptor (e.g., 'soil_map_unit', 'parcel')
        """
        self.feature_id = feature_id
        self.polygon_coords = polygon_coords
        self.components = components or []
        self.attributes = attributes or {}
        self.feature_type = feature_type or 'generic_polygon'
        self.centroid = self._calculate_centroid()
    
    def _calculate_centroid(self) -> Tuple[float, float]:
        """Calculate polygon centroid (simple average of vertices)."""
        if not self.polygon_coords:
            return (0.0, 0.0)
        lats = [c[0] for c in self.polygon_coords]
        lons = [c[1] for c in self.polygon_coords]
        return (sum(lats) / len(lats), sum(lons) / len(lons))
    
    def get_dominant_component(self) -> Optional[PolygonComponent]:
        """Get the component with highest percentage."""
        if not self.components:
            return None
        return max(self.components, key=lambda c: c.percentage)
    
    def get_weighted_attribute(self, 
                              attr_name: str,
                              aggregation: str = 'weighted_mean') -> Optional[float]:
        """Calculate component-weighted attribute value.
        
        Args:
            attr_name: Attribute name to aggregate
            aggregation: Aggregation method
                - 'weighted_mean': Percentage-weighted average
                - 'mean': Simple average across components
                - 'sum': Sum of values
                - 'max': Maximum value
                - 'min': Minimum value
                - 'median': Median value
        
        Returns:
            Aggregated value or None if attribute not found
        """
        if not self.components:
            return None
        
        values = []
        weights = []
        
        for comp in self.components:
            val = comp.get_attribute(attr_name)
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
    
    def get_attribute(self, attr_name: str, default: Any = None) -> Any:
        """Get polygon-level attribute."""
        return self.attributes.get(attr_name, default)
    
    def set_attribute(self, attr_name: str, value: Any) -> None:
        """Set polygon-level attribute."""
        self.attributes[attr_name] = value
    
    def get_component_diversity(self) -> float:
        """Calculate Shannon diversity index for components.
        
        Returns:
            Diversity index (0 = single component, higher = more diverse)
        """
        if not self.components:
            return 0.0
        
        total_pct = sum(c.percentage for c in self.components)
        if total_pct == 0:
            return 0.0
        
        diversity = 0.0
        for comp in self.components:
            if comp.percentage > 0:
                p = comp.percentage / total_pct
                diversity -= p * (p if p == 0 else statistics.log(p, 2))
        
        return diversity


####################################################################
# Generic Polygon Discretization Functions
####################################################################

def discretize_polygon_features(
    features: Sequence[PolygonFeature],
    level: int = 12,
    method: str = 'centroid',
    min_area: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """Discretize polygon features to DGGS cells.
    
    Two discretization methods:
    1. 'centroid': Assign polygon to cell containing its centroid (fast)
    2. 'coverage': Cover entire polygon with cells (accurate, slower)
    
    Args:
        features: List of PolygonFeature objects
        level: Target DGGS cell level (10-15)
        method: 'centroid' or 'coverage'
        min_area: Minimum polygon area (sq meters) to include
    
    Returns:
        Dictionary mapping cell_token -> {
            'feature_id': str,
            'feature_type': str,
            'num_components': int,
            'dominant_component': str,
            'centroid': (lat, lon),
            'num_cells': int (for coverage method)
        }
    
    Example:
        features = [
            PolygonFeature('poly1', [(40.0, -74.0), ...], components=[...])
        ]
        result = discretize_polygon_features(features, level=12, method='centroid')
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        # Optional area filter
        if min_area is not None:
            # Estimate area (simplified)
            if len(feature.polygon_coords) < 3:
                continue
        
        if method == 'centroid':
            lat, lon = feature.centroid
            cell_token = grid.latlon_to_token(lat, lon, level)
            
            dominant = feature.get_dominant_component()
            
            # Handle different component types (PolygonComponent or dict)
            if dominant is not None:
                if isinstance(dominant, PolygonComponent):
                    dominant_name = dominant.name
                    dominant_pct = dominant.percentage
                elif isinstance(dominant, dict):
                    dominant_name = dominant.get('series_name', dominant.get('name', str(dominant)))
                    dominant_pct = dominant.get('percentage', 0)
                else:
                    dominant_name = str(dominant)
                    dominant_pct = 0
            else:
                dominant_name = None
                dominant_pct = 0
            
            result[cell_token] = {
                'feature_id': feature.feature_id,
                'feature_type': feature.feature_type,
                'centroid': feature.centroid,
                'num_components': len(feature.components) if hasattr(feature.components, '__len__') else 0,
                'dominant_component': dominant_name,
                'dominant_percentage': dominant_pct,
                'component_diversity': feature.get_component_diversity()
            }
            
            # Include polygon-level attributes
            result[cell_token].update({
                f'polygon_{k}': v for k, v in feature.attributes.items()
            })
        
        elif method == 'coverage':
            # Use strict polygon discretization
            # Convert to GeoJSON format [[lon, lat], ...]
            geojson_coords = [[(c[1], c[0]) for c in feature.polygon_coords]]
            
            try:
                cell_tokens = discretize_polygon_strict(geojson_coords, level=level)
                
                for token in cell_tokens:
                    if token not in result:
                        result[token] = {
                            'feature_id': feature.feature_id,
                            'feature_type': feature.feature_type,
                            'components': feature.components,
                            'num_cells': 1,
                            'component_diversity': feature.get_component_diversity()
                        }
                    else:
                        # Multiple features in same cell
                        result[token]['num_cells'] = result[token].get('num_cells', 1) + 1
            
            except Exception as e:
                print(f"Error processing feature {feature.feature_id}: {e}")
    
    return result


def discretize_polygon_attributes(
    features: Sequence[PolygonFeature],
    attributes: List[str],
    level: int = 12,
    aggregation_funcs: Optional[Dict[str, str]] = None,
    weight_by_component: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Discretize polygon attributes with statistical aggregation.
    
    Aggregates component-level attributes to DGGS cells using various
    statistical methods. Supports weighted and unweighted aggregation.
    
    Args:
        features: List of PolygonFeature objects
        attributes: List of attribute names to aggregate
        level: Target DGGS cell level
        aggregation_funcs: Dict mapping attribute -> aggregation function
            Options: 'weighted_mean', 'mean', 'sum', 'max', 'min', 'median'
        weight_by_component: If True, weight by component percentage
    
    Returns:
        Dictionary mapping cell_token -> {
            'attr1_mean': value,
            'attr2_weighted_mean': value,
            ...
        }
    
    Example:
        features = [
            PolygonFeature('poly1', coords, components=[
                PolygonComponent('comp1', 60, {'pH': 6.5, 'depth': 100}),
                PolygonComponent('comp2', 40, {'pH': 7.0, 'depth': 80})
            ])
        ]
        
        result = discretize_polygon_attributes(
            features,
            attributes=['pH', 'depth'],
            aggregation_funcs={'pH': 'weighted_mean', 'depth': 'mean'}
        )
    """
    if aggregation_funcs is None:
        aggregation_funcs = {
            attr: 'weighted_mean' if weight_by_component else 'mean' 
            for attr in attributes
        }
    
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        lat, lon = feature.centroid
        cell_token = grid.latlon_to_token(lat, lon, level)
        
        if cell_token not in result:
            result[cell_token] = {
                'feature_id': feature.feature_id,
                'feature_type': feature.feature_type
            }
        
        # Aggregate each attribute
        for attr in attributes:
            agg_func = aggregation_funcs.get(attr, 'weighted_mean')
            value = feature.get_weighted_attribute(attr, aggregation=agg_func)
            
            if value is not None:
                result[cell_token][f'{attr}_{agg_func}'] = value
        
        # Store metadata
        result[cell_token]['num_components'] = len(feature.components)
    
    return result


def discretize_polygon_hierarchical(
    features: Sequence[PolygonFeature],
    hierarchy_levels: Dict[str, Any],
    attributes_per_level: Dict[str, List[str]],
    level: int = 12
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Discretize hierarchical component data.
    
    For polygons with multi-level component hierarchies (e.g., soil horizons,
    vegetation strata, building floors).
    
    Args:
        features: List of PolygonFeature objects
        hierarchy_levels: Dict defining hierarchy structure
        attributes_per_level: Dict mapping level name -> attribute list
        level: Target DGGS cell level
    
    Returns:
        Dict mapping cell_token -> {level_name -> {attr: value, ...}}
    
    Example:
        # Soil horizons
        hierarchy_levels = {
            'A_horizon': (0, 25),    # cm depth
            'B_horizon': (25, 100),
            'C_horizon': (100, 200)
        }
        
        attributes_per_level = {
            'A_horizon': ['clay_pct', 'organic_matter'],
            'B_horizon': ['clay_pct', 'bulk_density'],
            'C_horizon': ['parent_material']
        }
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for feature in features:
        lat, lon = feature.centroid
        cell_token = grid.latlon_to_token(lat, lon, level)
        
        if cell_token not in result:
            result[cell_token] = {}
        
        # Process each hierarchy level
        for level_name, level_criteria in hierarchy_levels.items():
            if level_name not in result[cell_token]:
                result[cell_token][level_name] = {}
            
            # Get attributes for this level
            attrs = attributes_per_level.get(level_name, [])
            
            # Find matching components
            matching_components = []
            for comp in feature.components:
                # Check if component matches level criteria
                # (Implementation depends on hierarchy type)
                matching_components.append(comp)
            
            # Aggregate attributes across matching components
            for attr in attrs:
                values = [
                    comp.get_attribute(attr) 
                    for comp in matching_components 
                    if comp.get_attribute(attr) is not None
                ]
                
                if values:
                    try:
                        numeric_values = [float(v) for v in values]
                        result[cell_token][level_name][attr] = statistics.mean(numeric_values)
                    except (ValueError, TypeError):
                        # Non-numeric attributes - take most common
                        result[cell_token][level_name][attr] = max(
                            set(values), key=values.count
                        )
    
    return result


def discretize_polygon_categorical(
    features: Sequence[PolygonFeature],
    category_attribute: str,
    level: int = 12,
    method: str = 'dominant'
) -> Dict[str, Dict[str, Any]]:
    """Discretize categorical attributes from polygon components.
    
    Useful for land use classification, soil taxonomy, habitat types, etc.
    
    Args:
        features: List of PolygonFeature objects
        category_attribute: Name of categorical attribute
        level: Target DGGS cell level
        method: How to handle multiple categories
            - 'dominant': Use most abundant category
            - 'all': List all categories with percentages
            - 'diversity': Calculate categorical diversity
    
    Returns:
        Dictionary with category information per cell
    
    Example:
        # Land use classification
        result = discretize_polygon_categorical(
            parcels,
            category_attribute='land_use',
            method='all'
        )
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for feature in features:
        lat, lon = feature.centroid
        cell_token = grid.latlon_to_token(lat, lon, level)
        
        if cell_token not in result:
            result[cell_token] = {'feature_id': feature.feature_id}
        
        # Collect categories from components
        categories = {}
        for comp in feature.components:
            cat = comp.get_attribute(category_attribute)
            if cat is not None:
                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += comp.percentage
        
        if method == 'dominant':
            if categories:
                dominant_cat = max(categories.items(), key=lambda x: x[1])
                result[cell_token]['category'] = dominant_cat[0]
                result[cell_token]['percentage'] = dominant_cat[1]
        
        elif method == 'all':
            result[cell_token]['categories'] = categories
        
        elif method == 'diversity':
            # Calculate Shannon diversity
            total = sum(categories.values())
            if total > 0:
                diversity = 0
                for pct in categories.values():
                    p = pct / total
                    if p > 0:
                        diversity -= p * statistics.log(p, 2)
                result[cell_token]['diversity'] = diversity
                result[cell_token]['num_categories'] = len(categories)
    
    return result


####################################################################
# Utility Functions
####################################################################

def create_polygon_feature_from_dict(data: Dict[str, Any]) -> PolygonFeature:
    """Create PolygonFeature from dictionary.
    
    Useful for loading from JSON, CSV, or databases.
    
    Args:
        data: Dictionary with keys:
            - 'feature_id': str
            - 'polygon_coords': List[(lat, lon)]
            - 'components': List[dict] (optional)
            - 'attributes': dict (optional)
            - 'feature_type': str (optional)
    
    Returns:
        PolygonFeature object
    """
    components = []
    for comp_data in data.get('components', []):
        comp = PolygonComponent(
            name=comp_data.get('name', ''),
            percentage=comp_data.get('percentage', 100.0),
            attributes=comp_data.get('attributes', {})
        )
        components.append(comp)
    
    return PolygonFeature(
        feature_id=data['feature_id'],
        polygon_coords=data['polygon_coords'],
        components=components,
        attributes=data.get('attributes', {}),
        feature_type=data.get('feature_type')
    )


def aggregate_multiple_features_per_cell(
    discretized_data: Dict[str, List[PolygonFeature]],
    aggregation_method: str = 'area_weighted'
) -> Dict[str, Dict[str, Any]]:
    """Aggregate multiple features that fall in the same DGGS cell.
    
    When using 'coverage' method, multiple features may occupy the same cell.
    This function handles the aggregation.
    
    Args:
        discretized_data: Dict mapping cell_token -> List[PolygonFeature]
        aggregation_method: How to combine features
            - 'area_weighted': Weight by polygon area
            - 'count': Just count features
            - 'union': Union of all components
    
    Returns:
        Aggregated data per cell
    """
    result = {}
    
    for cell_token, features_list in discretized_data.items():
        if aggregation_method == 'count':
            result[cell_token] = {
                'num_features': len(features_list),
                'feature_ids': [f.feature_id for f in features_list]
            }
        
        elif aggregation_method == 'union':
            all_components = []
            for feature in features_list:
                all_components.extend(feature.components)
            
            result[cell_token] = {
                'num_features': len(features_list),
                'total_components': len(all_components),
                'unique_component_names': len(set(c.name for c in all_components))
            }
    
    return result
