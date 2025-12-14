"""
Polyline Discretization Examples

Demonstrates generic polyline discretization for various line-based spatial data:
1. Road network (transportation)
2. River network (hydrology)
3. Power transmission lines (utilities)
4. Hiking trails (recreation)
5. Pipeline network (infrastructure)
6. Migration corridors (ecology)
7. Railway network (transportation)

Each example shows:
- Creating PolylineFeature objects
- Discretizing with different methods
- Aggregating attributes
- Analyzing network topology
"""

from Dggs.discretizer_polyline import (
    PolylineFeature,
    LineSegment,
    discretize_polyline_features,
    discretize_polyline_attributes,
    discretize_polyline_network,
    discretize_polyline_density,
    discretize_polyline_flow,
    calculate_polyline_sinuosity,
    create_polyline_feature_from_dict
)
import json
from pathlib import Path
import sys

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def example_1_road_network():
    """Example: Road network with traffic attributes."""
    print("\n" + "="*70)
    print("Example 1: Road Network Discretization")
    print("="*70)
    
    # Create road segments with attributes
    highway_segments = [
        LineSegment(
            segment_id='seg1',
            start_point=(40.0, -74.0),
            end_point=(40.05, -74.05),
            attributes={
                'speed_limit': 65,
                'lanes': 4,
                'traffic_volume': 5000,
                'surface_type': 'asphalt'
            },
            weight=1.0
        ),
        LineSegment(
            segment_id='seg2',
            start_point=(40.05, -74.05),
            end_point=(40.1, -74.1),
            attributes={
                'speed_limit': 55,
                'lanes': 3,
                'traffic_volume': 3500,
                'surface_type': 'asphalt'
            },
            weight=0.8
        )
    ]
    
    # Create road feature
    highway = PolylineFeature(
        feature_id='highway_101',
        coordinates=[(40.0, -74.0), (40.05, -74.05), (40.1, -74.1)],
        segments=highway_segments,
        feature_type='highway',
        is_directed=True,
        attributes={'road_name': 'Highway 101', 'jurisdiction': 'State'}
    )
    
    # Create arterial road
    arterial_segments = [
        LineSegment(
            segment_id='art1',
            start_point=(40.02, -73.98),
            end_point=(40.05, -74.02),
            attributes={
                'speed_limit': 45,
                'lanes': 2,
                'traffic_volume': 2000,
                'surface_type': 'asphalt'
            },
            weight=0.6
        )
    ]
    
    arterial = PolylineFeature(
        feature_id='arterial_5th',
        coordinates=[(40.02, -73.98), (40.05, -74.02)],
        segments=arterial_segments,
        feature_type='arterial',
        is_directed=True,
        attributes={'road_name': '5th Avenue'}
    )
    
    roads = [highway, arterial]
    
    # 1. Basic discretization
    print("\n1. Basic Road Discretization (interpolate method):")
    cells = discretize_polyline_features(
        roads,
        level=12,
        method='interpolate',
        step_km=0.5
    )
    
    print(f"   Total cells covered: {len(cells)}")
    for token, data in list(cells.items())[:3]:
        print(f"   Cell {token}:")
        print(f"      Features: {data['features']}")
        print(f"      Types: {data['feature_types']}")
        print(f"      Length: {data['total_length_km']:.2f} km")
    
    # 2. Attribute aggregation
    print("\n2. Road Attribute Aggregation:")
    attr_cells = discretize_polyline_attributes(
        roads,
        attributes=['speed_limit', 'lanes', 'traffic_volume'],
        level=12,
        aggregation_funcs={
            'speed_limit': 'length_weighted',
            'lanes': 'max',
            'traffic_volume': 'sum'
        }
    )
    
    print(f"   Cells with attributes: {len(attr_cells)}")
    for token, data in list(attr_cells.items())[:2]:
        print(f"   Cell {token}:")
        if 'speed_limit_length_weighted' in data:
            print(f"      Avg speed limit: {data['speed_limit_length_weighted']:.1f} mph")
        if 'lanes_max' in data:
            print(f"      Max lanes: {data['lanes_max']}")
        if 'traffic_volume_sum' in data:
            print(f"      Total traffic: {data['traffic_volume_sum']:.0f} vehicles/day")
    
    # 3. Road density
    print("\n3. Road Density Analysis:")
    density = discretize_polyline_density(
        roads,
        level=12,
        normalize_by_area=True
    )
    
    for token, data in list(density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Density: {data.get('density_km_per_km2', 0):.2f} km/km²")
        print(f"      Features: {data['feature_count']}")
    
    # 4. Traffic flow
    print("\n4. Traffic Flow Analysis:")
    flow = discretize_polyline_flow(
        roads,
        flow_attribute='traffic_volume',
        level=12
    )
    
    for token, data in list(flow.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Total flow: {data['total_flow']:.0f} vehicles/day")
        print(f"      Avg flow: {data['avg_flow']:.0f} vehicles/day")


def example_2_river_network():
    """Example: River network with hydrological attributes."""
    print("\n" + "="*70)
    print("Example 2: River Network Discretization")
    print("="*70)
    
    # Main river
    main_river_segments = [
        LineSegment(
            'reach1',
            (39.0, -75.0), (39.1, -75.1),
            attributes={'discharge_m3s': 150, 'width_m': 50, 'depth_m': 3.5},
            weight=1.0
        ),
        LineSegment(
            'reach2',
            (39.1, -75.1), (39.2, -75.15),
            attributes={'discharge_m3s': 200, 'width_m': 60, 'depth_m': 4.0},
            weight=1.0
        )
    ]
    
    main_river = PolylineFeature(
        'mississippi_main',
        [(39.0, -75.0), (39.1, -75.1), (39.2, -75.15)],
        segments=main_river_segments,
        feature_type='river',
        is_directed=True,
        attributes={'river_name': 'Mississippi River', 'stream_order': 7}
    )
    
    # Tributary
    tributary_segments = [
        LineSegment(
            'trib1',
            (39.05, -74.95), (39.1, -75.0),
            attributes={'discharge_m3s': 30, 'width_m': 15, 'depth_m': 1.5},
            weight=0.5
        )
    ]
    
    tributary = PolylineFeature(
        'tributary_1',
        [(39.05, -74.95), (39.1, -75.0)],
        segments=tributary_segments,
        feature_type='stream',
        is_directed=True,
        attributes={'river_name': 'Clear Creek', 'stream_order': 3}
    )
    
    rivers = [main_river, tributary]
    
    # 1. River discretization
    print("\n1. River Network Discretization:")
    cells = discretize_polyline_features(
        rivers,
        level=12,
        method='interpolate',
        step_km=1.0
    )
    
    print(f"   River cells: {len(cells)}")
    for token, data in list(cells.items())[:2]:
        print(f"   Cell {token}: {data['feature_types']}, "
              f"{data['total_length_km']:.2f} km")
    
    # 2. Hydrological attributes
    print("\n2. Hydrological Attributes:")
    hydro = discretize_polyline_attributes(
        rivers,
        attributes=['discharge_m3s', 'width_m', 'depth_m'],
        level=12,
        aggregation_funcs={
            'discharge_m3s': 'sum',  # Accumulate discharge
            'width_m': 'length_weighted',
            'depth_m': 'mean'
        }
    )
    
    for token, data in list(hydro.items())[:2]:
        print(f"   Cell {token}:")
        if 'discharge_m3s_sum' in data:
            print(f"      Total discharge: {data['discharge_m3s_sum']:.1f} m³/s")
        if 'width_m_length_weighted' in data:
            print(f"      Avg width: {data['width_m_length_weighted']:.1f} m")
    
    # 3. Network connectivity
    print("\n3. River Network Connectivity:")
    network = discretize_polyline_network(
        rivers,
        level=12,
        connectivity_threshold_km=0.5
    )
    
    nodes = {k: v for k, v in network.items() if v['is_node']}
    print(f"   Network nodes: {len(nodes)}")
    for token, data in list(nodes.items())[:2]:
        print(f"   {data['node_type'].upper()} at {token}: "
              f"{data['num_connections']} connections")
    
    # 4. Flow analysis
    print("\n4. River Flow Analysis:")
    flow = discretize_polyline_flow(
        rivers,
        flow_attribute='discharge_m3s',
        level=12
    )
    
    for token, data in list(flow.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Total flow: {data['total_flow']:.1f} m³/s")
        print(f"      Max flow: {data['max_flow']:.1f} m³/s")
    
    # 5. Sinuosity
    print("\n5. River Sinuosity:")
    for river in rivers:
        sinuosity = calculate_polyline_sinuosity(river)
        print(f"   {river.feature_id}: {sinuosity:.2f} "
              f"({'meandering' if sinuosity > 1.3 else 'relatively straight'})")


def example_3_power_lines():
    """Example: Power transmission lines."""
    print("\n" + "="*70)
    print("Example 3: Power Transmission Network")
    print("="*70)
    
    # High voltage transmission line
    hv_segments = [
        LineSegment(
            'tower1_2',
            (41.0, -73.0), (41.05, -73.05),
            attributes={'voltage_kv': 500, 'current_a': 2000, 'power_mw': 1000},
            weight=1.0
        ),
        LineSegment(
            'tower2_3',
            (41.05, -73.05), (41.1, -73.1),
            attributes={'voltage_kv': 500, 'current_a': 2000, 'power_mw': 1000},
            weight=1.0
        )
    ]
    
    hv_line = PolylineFeature(
        'hv_line_1',
        [(41.0, -73.0), (41.05, -73.05), (41.1, -73.1)],
        segments=hv_segments,
        feature_type='transmission_line',
        is_directed=True,
        attributes={'line_name': 'HV-Main-1', 'operator': 'GridCo'}
    )
    
    # Distribution line
    dist_segments = [
        LineSegment(
            'pole1_2',
            (41.02, -73.02), (41.04, -73.04),
            attributes={'voltage_kv': 11, 'current_a': 300, 'power_mw': 3.3},
            weight=0.5
        )
    ]
    
    dist_line = PolylineFeature(
        'dist_line_1',
        [(41.02, -73.02), (41.04, -73.04)],
        segments=dist_segments,
        feature_type='distribution_line',
        is_directed=True,
        attributes={'line_name': 'Dist-5', 'operator': 'LocalUtility'}
    )
    
    power_lines = [hv_line, dist_line]
    
    # 1. Power line discretization
    print("\n1. Power Line Coverage:")
    cells = discretize_polyline_features(
        power_lines,
        level=13,
        method='interpolate',
        step_km=0.25
    )
    
    print(f"   Cells with power lines: {len(cells)}")
    for token, data in list(cells.items())[:2]:
        print(f"   Cell {token}: {data['feature_types']}")
    
    # 2. Power capacity
    print("\n2. Power Transmission Capacity:")
    capacity = discretize_polyline_attributes(
        power_lines,
        attributes=['voltage_kv', 'power_mw'],
        level=13,
        aggregation_funcs={
            'voltage_kv': 'max',
            'power_mw': 'sum'
        }
    )
    
    for token, data in list(capacity.items())[:2]:
        print(f"   Cell {token}:")
        if 'voltage_kv_max' in data:
            print(f"      Max voltage: {data['voltage_kv_max']:.0f} kV")
        if 'power_mw_sum' in data:
            print(f"      Total capacity: {data['power_mw_sum']:.1f} MW")
    
    # 3. Grid density
    print("\n3. Power Grid Density:")
    density = discretize_polyline_density(
        power_lines,
        level=13,
        normalize_by_area=True
    )
    
    for token, data in list(density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Line density: {data.get('density_km_per_km2', 0):.2f} km/km²")


def example_4_hiking_trails():
    """Example: Hiking trail network."""
    print("\n" + "="*70)
    print("Example 4: Hiking Trail Network")
    print("="*70)
    
    # Mountain trail with elevation
    trail_segments = [
        LineSegment(
            'section1',
            (42.0, -72.0), (42.01, -72.01),
            attributes={'elevation_m': 500, 'difficulty': 'moderate', 'grade_pct': 5},
            weight=1.0
        ),
        LineSegment(
            'section2',
            (42.01, -72.01), (42.02, -72.02),
            attributes={'elevation_m': 650, 'difficulty': 'difficult', 'grade_pct': 15},
            weight=1.0
        ),
        LineSegment(
            'section3',
            (42.02, -72.02), (42.03, -72.03),
            attributes={'elevation_m': 800, 'difficulty': 'difficult', 'grade_pct': 12},
            weight=1.0
        )
    ]
    
    mountain_trail = PolylineFeature(
        'appalachian_section_5',
        [(42.0, -72.0), (42.01, -72.01), (42.02, -72.02), (42.03, -72.03)],
        segments=trail_segments,
        feature_type='hiking_trail',
        is_directed=False,
        attributes={
            'trail_name': 'Appalachian Trail - Section 5',
            'distance_km': 3.5,
            'estimated_time_hrs': 2.5
        }
    )
    
    trails = [mountain_trail]
    
    # 1. Trail discretization
    print("\n1. Trail Discretization:")
    cells = discretize_polyline_features(
        trails,
        level=14,  # Fine resolution for trails
        method='interpolate',
        step_km=0.1
    )
    
    print(f"   Trail cells: {len(cells)}")
    print(f"   Total trail length in cells: "
          f"{sum(d['total_length_km'] for d in cells.values()):.2f} km")
    
    # 2. Elevation and difficulty
    print("\n2. Trail Characteristics:")
    trail_attrs = discretize_polyline_attributes(
        trails,
        attributes=['elevation_m', 'grade_pct'],
        level=14,
        aggregation_funcs={
            'elevation_m': 'mean',
            'grade_pct': 'max'
        }
    )
    
    for token, data in list(trail_attrs.items())[:3]:
        print(f"   Cell {token}:")
        if 'elevation_m_mean' in data:
            print(f"      Elevation: {data['elevation_m_mean']:.0f} m")
        if 'grade_pct_max' in data:
            print(f"      Max grade: {data['grade_pct_max']:.0f}%")
    
    # 3. Sinuosity
    print("\n3. Trail Sinuosity:")
    sinuosity = calculate_polyline_sinuosity(mountain_trail)
    print(f"   Trail sinuosity: {sinuosity:.2f}")
    print(f"   (1.0 = straight, {sinuosity:.2f} = "
          f"{'winding/switchback' if sinuosity > 1.2 else 'relatively direct'})")


def example_5_pipeline_network():
    """Example: Pipeline infrastructure."""
    print("\n" + "="*70)
    print("Example 5: Pipeline Network")
    print("="*70)
    
    # Gas pipeline
    pipeline_segments = [
        LineSegment(
            'pipe_seg1',
            (38.0, -76.0), (38.1, -76.1),
            attributes={
                'diameter_inches': 36,
                'pressure_psi': 1000,
                'flow_rate_mcf': 500,
                'material': 'steel'
            },
            weight=1.0
        ),
        LineSegment(
            'pipe_seg2',
            (38.1, -76.1), (38.2, -76.2),
            attributes={
                'diameter_inches': 30,
                'pressure_psi': 900,
                'flow_rate_mcf': 450,
                'material': 'steel'
            },
            weight=1.0
        )
    ]
    
    gas_pipeline = PolylineFeature(
        'gas_main_1',
        [(38.0, -76.0), (38.1, -76.1), (38.2, -76.2)],
        segments=pipeline_segments,
        feature_type='gas_pipeline',
        is_directed=True,
        attributes={'operator': 'GasCo', 'year_installed': 2010}
    )
    
    pipelines = [gas_pipeline]
    
    # 1. Pipeline coverage
    print("\n1. Pipeline Coverage:")
    cells = discretize_polyline_features(
        pipelines,
        level=12,
        method='interpolate',
        step_km=0.5
    )
    
    print(f"   Pipeline cells: {len(cells)}")
    
    # 2. Pipeline capacity
    print("\n2. Pipeline Capacity:")
    capacity = discretize_polyline_attributes(
        pipelines,
        attributes=['diameter_inches', 'flow_rate_mcf', 'pressure_psi'],
        level=12,
        aggregation_funcs={
            'diameter_inches': 'length_weighted',
            'flow_rate_mcf': 'mean',
            'pressure_psi': 'mean'
        }
    )
    
    for token, data in list(capacity.items())[:2]:
        print(f"   Cell {token}:")
        if 'diameter_inches_length_weighted' in data:
            print(f"      Avg diameter: {data['diameter_inches_length_weighted']:.1f} inches")
        if 'flow_rate_mcf_mean' in data:
            print(f"      Flow rate: {data['flow_rate_mcf_mean']:.0f} Mcf/day")


def example_6_migration_corridors():
    """Example: Wildlife migration corridors."""
    print("\n" + "="*70)
    print("Example 6: Wildlife Migration Corridors")
    print("="*70)
    
    # Migration route
    corridor_segments = [
        LineSegment(
            'corridor1',
            (43.0, -110.0), (43.1, -110.1),
            attributes={
                'corridor_width_m': 500,
                'habitat_quality': 0.8,
                'usage_frequency': 'high'
            },
            weight=1.0
        ),
        LineSegment(
            'corridor2',
            (43.1, -110.1), (43.2, -110.2),
            attributes={
                'corridor_width_m': 300,
                'habitat_quality': 0.6,
                'usage_frequency': 'medium'
            },
            weight=0.7
        )
    ]
    
    elk_corridor = PolylineFeature(
        'elk_migration_1',
        [(43.0, -110.0), (43.1, -110.1), (43.2, -110.2)],
        segments=corridor_segments,
        feature_type='migration_corridor',
        is_directed=True,
        attributes={'species': 'elk', 'season': 'fall'}
    )
    
    corridors = [elk_corridor]
    
    # 1. Corridor mapping
    print("\n1. Migration Corridor Mapping:")
    cells = discretize_polyline_features(
        corridors,
        level=12,
        method='interpolate',
        step_km=0.5
    )
    
    print(f"   Corridor cells: {len(cells)}")
    
    # 2. Habitat quality
    print("\n2. Corridor Quality Assessment:")
    quality = discretize_polyline_attributes(
        corridors,
        attributes=['corridor_width_m', 'habitat_quality'],
        level=12,
        aggregation_funcs={
            'corridor_width_m': 'mean',
            'habitat_quality': 'weighted_mean'
        }
    )
    
    for token, data in list(quality.items())[:2]:
        print(f"   Cell {token}:")
        if 'corridor_width_m_mean' in data:
            print(f"      Width: {data['corridor_width_m_mean']:.0f} m")
        if 'habitat_quality_weighted_mean' in data:
            print(f"      Quality: {data['habitat_quality_weighted_mean']:.2f}")


def example_7_railway_network():
    """Example: Railway network with schedule data."""
    print("\n" + "="*70)
    print("Example 7: Railway Network")
    print("="*70)
    
    # Main railway line
    rail_segments = [
        LineSegment(
            'track1',
            (40.5, -74.5), (40.55, -74.55),
            attributes={
                'track_type': 'double',
                'max_speed_kmh': 160,
                'trains_per_day': 50,
                'electrified': True
            },
            weight=1.0
        ),
        LineSegment(
            'track2',
            (40.55, -74.55), (40.6, -74.6),
            attributes={
                'track_type': 'single',
                'max_speed_kmh': 120,
                'trains_per_day': 30,
                'electrified': True
            },
            weight=0.8
        )
    ]
    
    railway = PolylineFeature(
        'amtrak_northeast',
        [(40.5, -74.5), (40.55, -74.55), (40.6, -74.6)],
        segments=rail_segments,
        feature_type='railway',
        is_directed=False,
        attributes={'operator': 'Amtrak', 'route_name': 'Northeast Corridor'}
    )
    
    railways = [railway]
    
    # 1. Railway coverage
    print("\n1. Railway Network Coverage:")
    cells = discretize_polyline_features(
        railways,
        level=13,
        method='interpolate',
        step_km=0.25
    )
    
    print(f"   Railway cells: {len(cells)}")
    
    # 2. Railway characteristics
    print("\n2. Railway Characteristics:")
    rail_attrs = discretize_polyline_attributes(
        railways,
        attributes=['max_speed_kmh', 'trains_per_day'],
        level=13,
        aggregation_funcs={
            'max_speed_kmh': 'length_weighted',
            'trains_per_day': 'mean'
        }
    )
    
    for token, data in list(rail_attrs.items())[:2]:
        print(f"   Cell {token}:")
        if 'max_speed_kmh_length_weighted' in data:
            print(f"      Avg max speed: {data['max_speed_kmh_length_weighted']:.0f} km/h")
        if 'trains_per_day_mean' in data:
            print(f"      Daily trains: {data['trains_per_day_mean']:.0f}")
    
    # 3. Network density
    print("\n3. Railway Density:")
    density = discretize_polyline_density(
        railways,
        level=13,
        normalize_by_area=True
    )
    
    for token, data in list(density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Track density: {data.get('density_km_per_km2', 0):.2f} km/km²")


def example_8_dict_based_input():
    """Example: Creating polylines from dictionary (e.g., JSON, database)."""
    print("\n" + "="*70)
    print("Example 8: Dictionary-based Polyline Creation")
    print("="*70)
    
    # Road data from JSON/database
    road_data = {
        'feature_id': 'highway_95',
        'coordinates': [(35.0, -78.0), (35.1, -78.1), (35.2, -78.2)],
        'segments': [
            {
                'segment_id': 'seg1',
                'start_point': (35.0, -78.0),
                'end_point': (35.1, -78.1),
                'attributes': {'speed_limit': 70, 'lanes': 4},
                'weight': 1.0
            },
            {
                'segment_id': 'seg2',
                'start_point': (35.1, -78.1),
                'end_point': (35.2, -78.2),
                'attributes': {'speed_limit': 65, 'lanes': 3},
                'weight': 0.9
            }
        ],
        'attributes': {'road_name': 'I-95', 'jurisdiction': 'Interstate'},
        'feature_type': 'interstate',
        'is_directed': True
    }
    
    # Create feature from dict
    road = create_polyline_feature_from_dict(road_data)
    
    print(f"\n1. Created polyline from dictionary:")
    print(f"   Feature ID: {road.feature_id}")
    print(f"   Type: {road.feature_type}")
    print(f"   Length: {road.total_length_km:.2f} km")
    print(f"   Segments: {len(road.segments)}")
    
    # Discretize
    cells = discretize_polyline_features(
        [road],
        level=12,
        method='interpolate'
    )
    
    print(f"\n2. Discretization result:")
    print(f"   Cells covered: {len(cells)}")
    
    # Export to dict (for saving)
    export_data = {
        'feature_id': road.feature_id,
        'feature_type': road.feature_type,
        'total_length_km': road.total_length_km,
        'cells_covered': len(cells),
        'discretization_level': 12
    }
    
    print(f"\n3. Export data:")
    print(json.dumps(export_data, indent=2))


def run_all_examples():
    """Run all polyline discretization examples."""
    print("\n" + "#"*70)
    print("# POLYLINE DISCRETIZATION EXAMPLES")
    print("# Generic framework for line-based spatial data")
    print("#"*70)
    
    example_1_road_network()
    example_2_river_network()
    example_3_power_lines()
    example_4_hiking_trails()
    example_5_pipeline_network()
    example_6_migration_corridors()
    example_7_railway_network()
    example_8_dict_based_input()
    
    print("\n" + "#"*70)
    print("# All examples completed successfully!")
    print("#"*70 + "\n")


if __name__ == '__main__':
    run_all_examples()
