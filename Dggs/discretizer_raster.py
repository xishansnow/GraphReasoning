"""
DGGS Raster Discretization Module

Generic discretization functions for raster data (gridded spatial data).
This module provides general-purpose functions for discretizing any raster dataset
into DGGS cells, including categorical rasters (land cover, crop types) and 
continuous rasters (temperature, precipitation, elevation).

Raster data structure:
- Pixels: Regular grid with fixed resolution (e.g., 30m, 100m, 1km)
- Values: Can be categorical (land cover codes) or continuous (temperature)
- Attributes: Each pixel may have multiple attributes (value, confidence, timestamp)
- Temporal: Single-time or time-series rasters

Common raster types:
- Land cover/use: CDL, NLCD, ESA CCI, MODIS Land Cover
- Climate: PRISM, WorldClim, ERA5
- Elevation: SRTM, ASTER GDEM, USGS NED
- Remote sensing: Landsat, Sentinel, MODIS

Use cases:
- Spatial aggregation of raster data to hierarchical grids
- Multi-scale raster analysis
- Integration with vector data
- Temporal analysis of raster time series
- Attribute enrichment and derivation
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, Union
from dataclasses import dataclass
from s2sphere import CellId
from .dggs import DggsS2
import statistics
import math


####################################################################
# Generic Raster Data Models
####################################################################

@dataclass
class RasterPixel:
    """Generic raster pixel with spatial location and attributes.
    
    Represents a single pixel in a raster dataset with flexible attributes.
    """
    lat: float
    lon: float
    value: Any  # Primary value (can be categorical code or continuous value)
    attributes: Optional[Dict[str, Any]] = None  # Additional attributes
    timestamp: Optional[str] = None  # For temporal rasters
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'lat': self.lat,
            'lon': self.lon,
            'value': self.value,
            'attributes': self.attributes,
            'timestamp': self.timestamp
        }


@dataclass
class CategoricalPixel(RasterPixel):
    """Raster pixel with categorical value (land cover, crop type, etc.)."""
    category_name: Optional[str] = None  # Human-readable category name
    category_code: Optional[int] = None  # Numeric category code
    confidence: Optional[float] = None   # Classification confidence (0-1)
    
    def __post_init__(self):
        super().__post_init__()
        if self.category_code is None and isinstance(self.value, int):
            self.category_code = self.value


@dataclass
class ContinuousPixel(RasterPixel):
    """Raster pixel with continuous value (temperature, elevation, etc.)."""
    unit: Optional[str] = None          # Measurement unit
    precision: Optional[float] = None    # Measurement precision
    quality_flag: Optional[int] = None   # Data quality indicator
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.value, (int, float)):
            try:
                self.value = float(self.value)
            except (ValueError, TypeError):
                pass


####################################################################
# Generic Raster Discretization Functions
####################################################################

def discretize_raster_categorical(
    pixels: Sequence[Union[RasterPixel, CategoricalPixel]],
    level: int = 12,
    min_pixels: int = 1,
    value_attr: str = 'value',
    name_mapping: Optional[Dict[Any, str]] = None,
    category_mapping: Optional[Dict[Any, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Discretize categorical raster data to DGGS cells.
    
    Aggregates categorical raster pixels (e.g., land cover, crop types) into DGGS cells,
    computing class distribution, dominant category, and diversity metrics.
    
    Args:
        pixels: Sequence of RasterPixel or CategoricalPixel objects
        level: Target DGGS cell level (10-14 recommended)
        min_pixels: Minimum pixels required to create result for a cell
        value_attr: Attribute name containing the categorical value
        name_mapping: Optional dict mapping codes to human-readable names
        category_mapping: Optional dict mapping codes to category groups
    
    Returns:
        Dict mapping cell_token -> {
            'total_pixels': int,
            'total_area_m2': float,
            'total_area_acres': float,
            'categories': {category: {'count': int, 'percent': float, 'area_acres': float}},
            'dominant_category': {'name': str, 'code': Any, 'percent': float, 'area_acres': float},
            'diversity': float (0-1, Shannon diversity index),
            'timestamp': str (if temporal data)
        }
    
    Example:
        >>> pixels = [
        ...     CategoricalPixel(lat=40.0, lon=-100.0, value=1, category_name="Forest"),
        ...     CategoricalPixel(lat=40.0, lon=-100.01, value=2, category_name="Cropland")
        ... ]
        >>> result = discretize_raster_categorical(pixels, level=12)
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    # Pixel area: 30m x 30m = 900 m² (default, adjust if known)
    M2_TO_ACRES = 0.000247105
    pixel_area_m2 = 900  # Default 30m resolution
    
    for pixel in pixels:
        cell_token = grid.latlon_to_token(pixel.lat, pixel.lon, level)
        
        # Get pixel value
        if hasattr(pixel, value_attr):
            pixel_value = getattr(pixel, value_attr)
        elif isinstance(pixel, dict):
            pixel_value = pixel.get(value_attr)
        else:
            pixel_value = pixel.value
        
        # Get category name if available
        if isinstance(pixel, CategoricalPixel) and pixel.category_name:
            category_name = pixel.category_name
        elif name_mapping and pixel_value in name_mapping:
            category_name = name_mapping[pixel_value]
        else:
            category_name = str(pixel_value)
        
        # Initialize cell data
        if cell_token not in result:
            result[cell_token] = {
                'total_pixels': 0,
                'category_counts': {},
                'timestamp': getattr(pixel, 'timestamp', None)
            }
        
        result[cell_token]['total_pixels'] += 1
        
        # Track category counts
        if category_name not in result[cell_token]['category_counts']:
            result[cell_token]['category_counts'][category_name] = {
                'code': pixel_value,
                'count': 0,
                'pixels': []
            }
        
        result[cell_token]['category_counts'][category_name]['count'] += 1
        result[cell_token]['category_counts'][category_name]['pixels'].append(pixel.to_dict())
    
    # Convert to final format with statistics
    final_result = {}
    for cell_token, data in result.items():
        if data['total_pixels'] < min_pixels:
            continue
        
        total_area_m2 = data['total_pixels'] * pixel_area_m2
        total_area_acres = total_area_m2 * M2_TO_ACRES
        
        categories = {}
        max_count = 0
        dominant_category = None
        
        # Calculate category statistics
        for cat_name, cat_data in data['category_counts'].items():
            count = cat_data['count']
            percent = (count / data['total_pixels']) * 100
            area_acres = (count * pixel_area_m2) * M2_TO_ACRES
            
            categories[cat_name] = {
                'code': cat_data['code'],
                'count': count,
                'percent': round(percent, 2),
                'area_acres': round(area_acres, 2)
            }
            
            if count > max_count:
                max_count = count
                dominant_category = {
                    'name': cat_name,
                    'code': cat_data['code'],
                    'percent': round(percent, 2),
                    'area_acres': round(area_acres, 2)
                }
        
        # Calculate Shannon diversity index
        diversity = 0.0
        if len(categories) > 1:
            for cat_data in categories.values():
                p = cat_data['percent'] / 100
                if p > 0:
                    diversity -= p * math.log(p)
            # Normalize to 0-1 range
            max_diversity = math.log(len(categories))
            diversity = diversity / max_diversity if max_diversity > 0 else 0.0
        
        final_result[cell_token] = {
            'total_pixels': data['total_pixels'],
            'total_area_m2': total_area_m2,
            'total_area_acres': round(total_area_acres, 2),
            'categories': categories,
            'dominant_category': dominant_category,
            'category_diversity': round(diversity, 2),
            'timestamp': data['timestamp']
        }
    
    return final_result


def discretize_raster_continuous(
    pixels: Sequence[Union[RasterPixel, ContinuousPixel]],
    level: int = 12,
    min_pixels: int = 1,
    value_attr: str = 'value',
    aggregation_func: str = 'mean',
    custom_aggregator: Optional[Callable] = None
) -> Dict[str, Dict[str, Any]]:
    """Discretize continuous raster data to DGGS cells.
    
    Aggregates continuous raster pixels (e.g., temperature, elevation) into DGGS cells,
    computing statistics like mean, min, max, std deviation.
    
    Args:
        pixels: Sequence of RasterPixel or ContinuousPixel objects
        level: Target DGGS cell level
        min_pixels: Minimum pixels required to create result for a cell
        value_attr: Attribute name containing the continuous value
        aggregation_func: 'mean', 'median', 'min', 'max', 'sum', or 'custom'
        custom_aggregator: Custom aggregation function if aggregation_func='custom'
    
    Returns:
        Dict mapping cell_token -> {
            'total_pixels': int,
            'value': float (aggregated value),
            'mean': float,
            'median': float,
            'min': float,
            'max': float,
            'std': float,
            'sum': float,
            'unit': str (if available),
            'timestamp': str (if temporal data)
        }
    
    Example:
        >>> pixels = [
        ...     ContinuousPixel(lat=40.0, lon=-100.0, value=25.5, unit="celsius"),
        ...     ContinuousPixel(lat=40.0, lon=-100.01, value=26.0, unit="celsius")
        ... ]
        >>> result = discretize_raster_continuous(pixels, level=12, aggregation_func='mean')
    """
    grid = DGGSS2(level=level)
    result: Dict[str, Dict[str, Any]] = {}
    
    for pixel in pixels:
        cell_token = grid.latlon_to_token(pixel.lat, pixel.lon, level)
        
        # Get pixel value
        if hasattr(pixel, value_attr):
            pixel_value = getattr(pixel, value_attr)
        elif isinstance(pixel, dict):
            pixel_value = pixel.get(value_attr)
        else:
            pixel_value = pixel.value
        
        # Convert to float
        try:
            pixel_value = float(pixel_value)
        except (ValueError, TypeError):
            continue  # Skip invalid values
        
        # Initialize cell data
        if cell_token not in result:
            result[cell_token] = {
                'values': [],
                'unit': getattr(pixel, 'unit', None) if isinstance(pixel, ContinuousPixel) else None,
                'timestamp': getattr(pixel, 'timestamp', None)
            }
        
        result[cell_token]['values'].append(pixel_value)
    
    # Calculate statistics for each cell
    final_result = {}
    for cell_token, data in result.items():
        values = data['values']
        if len(values) < min_pixels:
            continue
        
        # Calculate statistics
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        sum_val = sum(values)
        
        # Apply aggregation function
        if aggregation_func == 'mean':
            aggregated_value = mean_val
        elif aggregation_func == 'median':
            aggregated_value = median_val
        elif aggregation_func == 'min':
            aggregated_value = min_val
        elif aggregation_func == 'max':
            aggregated_value = max_val
        elif aggregation_func == 'sum':
            aggregated_value = sum_val
        elif aggregation_func == 'custom' and custom_aggregator:
            aggregated_value = custom_aggregator(values)
        else:
            aggregated_value = mean_val
        
        final_result[cell_token] = {
            'total_pixels': len(values),
            'value': round(aggregated_value, 4),
            'mean': round(mean_val, 4),
            'median': round(median_val, 4),
            'min': round(min_val, 4),
            'max': round(max_val, 4),
            'std': round(std_val, 4),
            'sum': round(sum_val, 4),
            'unit': data['unit'],
            'timestamp': data['timestamp']
        }
    
    return final_result


def discretize_raster_temporal(
    pixels_by_time: Dict[str, Sequence[RasterPixel]],
    level: int = 12,
    categorical: bool = True,
    **kwargs
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Discretize temporal raster data (time series) to DGGS cells.
    
    Processes multiple time steps of raster data, returning discretized results
    for each time period.
    
    Args:
        pixels_by_time: Dict mapping timestamp -> list of pixels
        level: Target DGGS cell level
        categorical: True for categorical rasters, False for continuous
        **kwargs: Additional arguments passed to discretization function
    
    Returns:
        Dict mapping timestamp -> cell_token -> discretized data
    
    Example:
        >>> pixels_2020 = [CategoricalPixel(lat=40.0, lon=-100.0, value=1)]
        >>> pixels_2021 = [CategoricalPixel(lat=40.0, lon=-100.0, value=2)]
        >>> result = discretize_raster_temporal({
        ...     '2020': pixels_2020,
        ...     '2021': pixels_2021
        ... }, level=12, categorical=True)
    """
    result = {}
    
    discretize_func = discretize_raster_categorical if categorical else discretize_raster_continuous
    
    for timestamp, pixels in pixels_by_time.items():
        result[timestamp] = discretize_func(pixels, level=level, **kwargs)
    
    return result


def calculate_raster_change(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    categorical: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Calculate change between two discretized raster time periods.
    
    Args:
        before: Discretized raster data for earlier time period
        after: Discretized raster data for later time period
        categorical: True for categorical change, False for continuous
    
    Returns:
        Dict mapping cell_token -> {
            'before': value/category,
            'after': value/category,
            'changed': bool,
            'change_type': str (for categorical: 'stable', 'transition'),
            'change_value': float (for continuous: difference)
        }
    """
    result = {}
    all_cells = set(before.keys()) | set(after.keys())
    
    for cell_token in all_cells:
        before_data = before.get(cell_token, {})
        after_data = after.get(cell_token, {})
        
        if categorical:
            before_cat = before_data.get('dominant_category', {}).get('name')
            after_cat = after_data.get('dominant_category', {}).get('name')
            
            result[cell_token] = {
                'before': before_cat,
                'after': after_cat,
                'changed': before_cat != after_cat,
                'change_type': 'transition' if before_cat != after_cat else 'stable'
            }
        else:
            before_val = before_data.get('value', 0)
            after_val = after_data.get('value', 0)
            
            result[cell_token] = {
                'before': before_val,
                'after': after_val,
                'changed': abs(before_val - after_val) > 0.001,
                'change_value': after_val - before_val,
                'change_percent': ((after_val - before_val) / before_val * 100) if before_val != 0 else 0
            }
    
    return result
