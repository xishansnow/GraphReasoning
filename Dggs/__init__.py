"""
DGGS Package: Discrete Global Grid System

This package provides:
- Spatial utilities (spatial_utils.py): Core aggregation and geometric algorithms
- Raster discretization (raster.py): Generic discretization for raster/gridded data
- Polygon discretization (polygon.py): Generic discretization for polygon/vector data
- Polyline discretization (polyline.py): Generic discretization for polyline/line data
- Point discretization (point.py): Generic discretization for point/location data
- Topology enrichment (topo_enrichment.py): Graph construction with spatial relationships
"""

from .dggs import DggsS2

# Core spatial utilities (new consolidated module)
from .spatial_utils import (
    # Statistical aggregation
    aggregate_values,
    discretize_aggregate,
    discretize_weighted_aggregate,
    # Geometric algorithms
    point_in_polygon,
    point_in_ring,
    segments_intersect,
    interpolate_line_segment,
    discretize_polygon_strict,
    # Utilities
    calculate_distance_km,
    calculate_bearing,
    calculate_centroid,
    calculate_bbox,
)

# Generic raster discretization
from .discretizer_raster import (
    RasterPixel,
    CategoricalPixel,
    ContinuousPixel,
    discretize_raster_categorical,
    discretize_raster_continuous,
    discretize_raster_temporal,
    calculate_raster_change,
)

# Generic polygon discretization
from .discretizer_polygon import (
    PolygonFeature,
    PolygonComponent,
    discretize_polygon_features,
    discretize_polygon_attributes,
    discretize_polygon_hierarchical,
    discretize_polygon_categorical,
)

# Generic polyline discretization
from .discretizer_polyline import (
    PolylineFeature,
    LineSegment,
    discretize_polyline_features,
    discretize_polyline_attributes,
    discretize_polyline_network,
    discretize_polyline_density,
    discretize_polyline_flow,
    calculate_polyline_sinuosity,
    create_polyline_feature_from_dict,
)

# Generic point discretization
from .discretizer_point import (
    PointFeature,
    PointCluster,
    discretize_point_features,
    discretize_point_attributes,
    discretize_point_density,
    discretize_point_clusters,
    discretize_point_patterns,
    discretize_point_temporal,
    create_point_feature_from_dict,
    spatial_cluster_points,
    calculate_point_statistics,
    filter_points_by_attribute,
)

# Topology enrichment
from .topo_enrichment import build_topology_enriched_graph

# Discretized data to knowledge graph conversion
# 注意: discretized_cdl_to_triplets 已移至 examples/raster_examples.py
#       discretized_ssurgo_to_triplets 已移至 examples/polygon_examples.py
from .discretized_to_kg import (
    SpatialEntity,
    SpatialRelationship,
    discretized_agricultural_intensity_to_triplets,
    spatial_adjacency_to_triplets,
    temporal_triplets,
    create_knowledge_graph_from_discretized_data,
    triplets_to_dataframe,
    merge_into_existing_graph,
    export_triplets_to_csv,
    export_triplets_to_json,
    export_graph_to_graphml,
    export_graph_to_rdf_turtle,
    prepare_for_graph_reasoning,
)

__all__ = [
    "DGGSS2",
    # Value-based discretization
    "discretize_direct_assignment",
    "discretize_aggregate",
    "discretize_multiscale",
    "discretize_weighted_aggregate",
    "discretize_interpolate",
    # Geometric discretization
    "discretize_points",
    "discretize_paths",
    "discretize_regions_bbox",
    "discretize_buffers",
    "discretize_geojson",
    "discretize_polygon_strict",
    # Generic raster discretization
    "RasterPixel",
    "CategoricalPixel",
    "ContinuousPixel",
    "discretize_raster_categorical",
    "discretize_raster_continuous",
    "discretize_raster_temporal",
    "calculate_raster_change",
    # Generic polygon discretization
    "PolygonFeature",
    "PolygonComponent",
    "discretize_polygon_features",
    "discretize_polygon_attributes",
    "discretize_polygon_hierarchical",
    "discretize_polygon_categorical",
    # Generic polyline discretization
    "PolylineFeature",
    "LineSegment",
    "discretize_polyline_features",
    "discretize_polyline_attributes",
    "discretize_polyline_network",
    "discretize_polyline_density",
    "discretize_polyline_flow",
    "calculate_polyline_sinuosity",
    "create_polyline_feature_from_dict",
    # Generic point discretization
    "PointFeature",
    "PointCluster",
    "discretize_point_features",
    "discretize_point_attributes",
    "discretize_point_density",
    "discretize_point_clusters",
    "discretize_point_patterns",
    "discretize_point_temporal",
    "create_point_feature_from_dict",
    "spatial_cluster_points",
    "calculate_point_statistics",
    "filter_points_by_attribute",
    # Topology enrichment
    "build_topology_enriched_graph",
    # Knowledge graph conversion
    # 注意: discretized_cdl_to_triplets 和 discretized_ssurgo_to_triplets 
    #       已移至 examples/raster_examples.py 和 examples/polygon_examples.py
    "SpatialEntity",
    "SpatialRelationship",
    "discretized_agricultural_intensity_to_triplets",
    "spatial_adjacency_to_triplets",
    "temporal_triplets",
    "create_knowledge_graph_from_discretized_data",
    "triplets_to_dataframe",
    "merge_into_existing_graph",
    "export_triplets_to_csv",
    "export_triplets_to_json",
    "export_graph_to_graphml",
    "export_graph_to_rdf_turtle",
    "prepare_for_graph_reasoning",
]

# ========================================
# Legacy Modules Removed
# ========================================
# The legacy modules (discretize.py and geometry.py) have been consolidated
# into spatial_utils.py. All functionality is now available from spatial_utils.py
# and the main DGGS exports above.
#
# Migration Guide:
#  OLD: from Dggs.discretize import discretize_aggregate
#  NEW: from Dggs import discretize_aggregate
#
#  OLD: from Dggs.geometry import discretize_polygon_strict
#  NEW: from Dggs import discretize_polygon_strict
#
#  OLD: from Dggs.geometry import discretize_points
#  NEW: This function is now in spatial_utils.py (use sparingly, mostly superseded)
