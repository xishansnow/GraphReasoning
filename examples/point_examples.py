"""
Point Discretization Examples

Demonstrates generic point discretization for various point-based spatial data:
1. Weather stations (environmental monitoring)
2. Points of Interest (POIs) - restaurants
3. Soil sampling sites (field surveys)
4. Cell towers (infrastructure)
5. Crime incidents (events with timestamps)
6. Hospital locations (facilities)
7. Tree inventory (natural features)
8. Air quality sensors (continuous monitoring)

Each example shows:
- Creating PointFeature objects
- Discretizing with different methods
- Aggregating attributes
- Analyzing spatial patterns
"""

from pathlib import Path
import sys

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Dggs.discretizer_point import (
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
    filter_points_by_attribute
)
import json


def example_1_weather_stations():
    """Example: Weather station network with meteorological data."""
    print("\n" + "="*70)
    print("Example 1: Weather Station Network")
    print("="*70)
    
    # Create weather stations
    stations = [
        PointFeature(
            'station_001', 40.0, -74.0,
            attributes={'temperature': 25.5, 'humidity': 65, 'pressure': 1013},
            feature_type='weather_station',
            weight=1.0
        ),
        PointFeature(
            'station_002', 40.05, -74.05,
            attributes={'temperature': 26.2, 'humidity': 62, 'pressure': 1012},
            feature_type='weather_station',
            weight=1.0
        ),
        PointFeature(
            'station_003', 40.1, -74.1,
            attributes={'temperature': 24.8, 'humidity': 68, 'pressure': 1014},
            feature_type='weather_station',
            weight=1.0
        ),
        PointFeature(
            'station_004', 40.02, -74.02,
            attributes={'temperature': 25.8, 'humidity': 64, 'pressure': 1013},
            feature_type='weather_station',
            weight=1.2  # Higher quality station
        )
    ]
    
    # 1. Basic point discretization
    print("\n1. Basic Station Discretization:")
    cells = discretize_point_features(stations, level=12)
    
    print(f"   Total cells with stations: {len(cells)}")
    for token, data in list(cells.items())[:3]:
        print(f"   Cell {token}:")
        print(f"      Stations: {data['features']}")
        print(f"      Count: {data['num_points']}")
    
    # 2. Attribute aggregation
    print("\n2. Meteorological Data Aggregation:")
    weather_attrs = discretize_point_attributes(
        stations,
        attributes=['temperature', 'humidity', 'pressure'],
        aggregation_funcs={
            'temperature': 'weighted_mean',
            'humidity': 'mean',
            'pressure': 'mean'
        }
    )
    
    print(f"   Cells with data: {len(weather_attrs)}")
    for token, data in list(weather_attrs.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Avg temp: {data.get('temperature_weighted_mean', 0):.1f}°C")
        print(f"      Avg humidity: {data.get('humidity_mean', 0):.1f}%")
        print(f"      Avg pressure: {data.get('pressure_mean', 0):.1f} hPa")
    
    # 3. Station density
    print("\n3. Station Coverage Density:")
    density = discretize_point_density(stations, level=12, normalize_by_area=True)
    
    for token, data in list(density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Stations: {data['point_count']}")
        print(f"      Density: {data.get('density_points_per_km2', 0):.4f} stations/km²")
    
    # 4. Calculate statistics
    print("\n4. Temperature Statistics:")
    temp_stats = calculate_point_statistics(stations, 'temperature')
    print(f"   Mean: {temp_stats['mean']:.2f}°C")
    print(f"   Std Dev: {temp_stats['std']:.2f}°C")
    print(f"   Range: {temp_stats['min']:.2f}°C - {temp_stats['max']:.2f}°C")
    
    # 5. Filter stations
    print("\n5. High Temperature Stations:")
    hot_stations = filter_points_by_attribute(
        stations,
        'temperature',
        lambda t: t > 25.5
    )
    print(f"   Stations with temp > 25.5°C: {len(hot_stations)}")
    for station in hot_stations:
        print(f"      {station.feature_id}: {station.get_attribute('temperature')}°C")


def example_2_points_of_interest():
    """Example: Restaurant POIs with ratings."""
    print("\n" + "="*70)
    print("Example 2: Restaurant POIs (Points of Interest)")
    print("="*70)
    
    # Create restaurant POIs
    restaurants = [
        PointFeature(
            'rest_001', 40.75, -73.99,
            attributes={'rating': 4.5, 'price_level': 3, 'cuisine': 'italian'},
            feature_type='restaurant',
            cluster_id='downtown'
        ),
        PointFeature(
            'rest_002', 40.76, -73.98,
            attributes={'rating': 4.2, 'price_level': 2, 'cuisine': 'chinese'},
            feature_type='restaurant',
            cluster_id='downtown'
        ),
        PointFeature(
            'rest_003', 40.75, -73.97,
            attributes={'rating': 4.8, 'price_level': 4, 'cuisine': 'french'},
            feature_type='restaurant',
            cluster_id='downtown'
        ),
        PointFeature(
            'rest_004', 40.80, -74.00,
            attributes={'rating': 3.9, 'price_level': 2, 'cuisine': 'mexican'},
            feature_type='restaurant',
            cluster_id='uptown'
        ),
        PointFeature(
            'rest_005', 40.81, -74.01,
            attributes={'rating': 4.3, 'price_level': 3, 'cuisine': 'japanese'},
            feature_type='restaurant',
            cluster_id='uptown'
        )
    ]
    
    # 1. POI discretization
    print("\n1. Restaurant Distribution:")
    cells = discretize_point_features(restaurants, level=13)
    
    print(f"   Cells with restaurants: {len(cells)}")
    for token, data in list(cells.items())[:3]:
        print(f"   Cell {token}: {data['num_points']} restaurant(s)")
    
    # 2. Rating and price aggregation
    print("\n2. Restaurant Quality Metrics:")
    poi_attrs = discretize_point_attributes(
        restaurants,
        attributes=['rating', 'price_level'],
        aggregation_funcs={
            'rating': 'mean',
            'price_level': 'median'
        }
    )
    
    for token, data in list(poi_attrs.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Avg rating: {data.get('rating_mean', 0):.1f} stars")
        print(f"      Median price: {'$' * int(data.get('price_level_median', 0))}")
    
    # 3. Cluster analysis
    print("\n3. Neighborhood Clustering:")
    clusters = discretize_point_clusters(
        restaurants,
        level=12,
        cluster_by='cluster_id',
        aggregate_attributes=['rating']
    )
    
    for token, data in list(clusters.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Clusters: {list(data['clusters'].keys())}")
        print(f"      Dominant: {data.get('dominant_cluster', 'N/A')}")
        print(f"      Diversity: {data.get('cluster_diversity', 0):.2f}")
    
    # 4. Restaurant density
    print("\n4. Restaurant Density Analysis:")
    density = discretize_point_density(restaurants, level=13, normalize_by_area=True)
    
    high_density = {k: v for k, v in density.items() if v['point_count'] >= 2}
    print(f"   High-density cells: {len(high_density)}")
    for token, data in list(high_density.items())[:2]:
        print(f"   Cell {token}: {data['point_count']} restaurants, "
              f"{data.get('density_points_per_km2', 0):.2f}/km²")


def example_3_soil_sampling():
    """Example: Soil sampling sites with chemical properties."""
    print("\n" + "="*70)
    print("Example 3: Soil Sampling Sites")
    print("="*70)
    
    # Create sampling sites
    samples = [
        PointFeature(
            'sample_001', 39.5, -75.5,
            attributes={'pH': 6.5, 'nitrogen': 25, 'phosphorus': 15, 'organic_matter': 3.2},
            feature_type='soil_sample',
            cluster_id='field_A'
        ),
        PointFeature(
            'sample_002', 39.51, -75.51,
            attributes={'pH': 6.8, 'nitrogen': 28, 'phosphorus': 18, 'organic_matter': 3.5},
            feature_type='soil_sample',
            cluster_id='field_A'
        ),
        PointFeature(
            'sample_003', 39.52, -75.49,
            attributes={'pH': 6.2, 'nitrogen': 22, 'phosphorus': 12, 'organic_matter': 2.8},
            feature_type='soil_sample',
            cluster_id='field_A'
        ),
        PointFeature(
            'sample_004', 39.55, -75.48,
            attributes={'pH': 7.0, 'nitrogen': 30, 'phosphorus': 20, 'organic_matter': 4.0},
            feature_type='soil_sample',
            cluster_id='field_B'
        )
    ]
    
    # 1. Sample distribution
    print("\n1. Sampling Coverage:")
    cells = discretize_point_features(samples, level=13)
    
    print(f"   Cells sampled: {len(cells)}")
    for token, data in cells.items():
        print(f"   Cell {token}: {data['num_points']} sample(s)")
    
    # 2. Soil property aggregation
    print("\n2. Soil Chemical Properties:")
    soil_props = discretize_point_attributes(
        samples,
        attributes=['pH', 'nitrogen', 'phosphorus', 'organic_matter'],
        aggregation_funcs={
            'pH': 'mean',
            'nitrogen': 'mean',
            'phosphorus': 'mean',
            'organic_matter': 'mean'
        }
    )
    
    for token, data in list(soil_props.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      pH: {data.get('pH_mean', 0):.1f}")
        print(f"      N: {data.get('nitrogen_mean', 0):.1f} ppm")
        print(f"      P: {data.get('phosphorus_mean', 0):.1f} ppm")
        print(f"      OM: {data.get('organic_matter_mean', 0):.1f}%")
    
    # 3. Field-based clustering
    print("\n3. Field-based Analysis:")
    field_clusters = discretize_point_clusters(
        samples,
        level=12,
        cluster_by='cluster_id',
        aggregate_attributes=['pH', 'nitrogen']
    )
    
    for token, data in field_clusters.items():
        print(f"   Cell {token}:")
        print(f"      Fields: {list(data['clusters'].keys())}")
        if 'cluster_attributes' in data:
            for field, attrs in data['cluster_attributes'].items():
                print(f"      {field}: pH={attrs.get('pH', {}).get('mean', 0):.1f}, "
                      f"N={attrs.get('nitrogen', {}).get('mean', 0):.1f}")
    
    # 4. Spatial patterns
    print("\n4. Sampling Pattern Analysis:")
    patterns = discretize_point_patterns(
        samples,
        level=12,
        pattern_metrics=['centroid', 'spread', 'nearest_neighbor']
    )
    
    for token, data in patterns.items():
        print(f"   Cell {token}:")
        print(f"      Centroid: {data.get('centroid', (0, 0))}")
        print(f"      Spread: {data.get('spread_km', 0):.2f} km")
        if 'avg_nearest_neighbor_km' in data:
            print(f"      Avg NN distance: {data['avg_nearest_neighbor_km']:.2f} km")


def example_4_cell_towers():
    """Example: Cell tower infrastructure."""
    print("\n" + "="*70)
    print("Example 4: Cell Tower Infrastructure")
    print("="*70)
    
    # Create cell towers
    towers = [
        PointFeature(
            'tower_001', 41.0, -74.5,
            attributes={'carrier': 'Verizon', 'frequency': 2100, 'power': 50},
            feature_type='cell_tower'
        ),
        PointFeature(
            'tower_002', 41.05, -74.55,
            attributes={'carrier': 'AT&T', 'frequency': 1900, 'power': 45},
            feature_type='cell_tower'
        ),
        PointFeature(
            'tower_003', 41.1, -74.6,
            attributes={'carrier': 'T-Mobile', 'frequency': 2100, 'power': 48},
            feature_type='cell_tower'
        ),
        PointFeature(
            'tower_004', 41.02, -74.52,
            attributes={'carrier': 'Verizon', 'frequency': 1900, 'power': 52},
            feature_type='cell_tower'
        )
    ]
    
    # 1. Tower distribution
    print("\n1. Tower Coverage:")
    cells = discretize_point_features(towers, level=12)
    
    print(f"   Cells with towers: {len(cells)}")
    for token, data in cells.items():
        print(f"   Cell {token}: {data['num_points']} tower(s)")
    
    # 2. Infrastructure density
    print("\n2. Infrastructure Density:")
    density = discretize_point_density(
        towers,
        level=12,
        normalize_by_area=True,
        kernel_radius_cells=1  # Include neighbors for smoother density
    )
    
    for token, data in list(density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Towers: {data['point_count']}")
        if 'kernel_count' in data:
            print(f"      Kernel count (with neighbors): {data['kernel_count']}")
        print(f"      Density: {data.get('density_points_per_km2', 0):.4f} towers/km²")
    
    # 3. Signal coverage metrics
    print("\n3. Signal Coverage Metrics:")
    coverage = discretize_point_attributes(
        towers,
        attributes=['frequency', 'power'],
        aggregation_funcs={
            'frequency': 'mean',
            'power': 'max'
        }
    )
    
    for token, data in list(coverage.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Avg frequency: {data.get('frequency_mean', 0):.0f} MHz")
        print(f"      Max power: {data.get('power_max', 0):.0f} W")


def example_5_crime_incidents():
    """Example: Crime incidents with timestamps."""
    print("\n" + "="*70)
    print("Example 5: Crime Incident Analysis")
    print("="*70)
    
    # Create crime incidents
    incidents = [
        PointFeature(
            'crime_001', 40.7, -73.9,
            attributes={'type': 'theft', 'severity': 2},
            feature_type='crime',
            timestamp='2024-01-15T14:30:00',
            cluster_id='hotspot_1'
        ),
        PointFeature(
            'crime_002', 40.71, -73.91,
            attributes={'type': 'assault', 'severity': 4},
            feature_type='crime',
            timestamp='2024-01-16T22:15:00',
            cluster_id='hotspot_1'
        ),
        PointFeature(
            'crime_003', 40.72, -73.89,
            attributes={'type': 'theft', 'severity': 2},
            feature_type='crime',
            timestamp='2024-01-17T18:45:00',
            cluster_id='hotspot_1'
        ),
        PointFeature(
            'crime_004', 40.75, -73.95,
            attributes={'type': 'vandalism', 'severity': 1},
            feature_type='crime',
            timestamp='2024-01-18T03:20:00',
            cluster_id='hotspot_2'
        ),
        PointFeature(
            'crime_005', 40.73, -73.90,
            attributes={'type': 'theft', 'severity': 3},
            feature_type='crime',
            timestamp='2024-01-19T12:00:00',
            cluster_id='hotspot_1'
        )
    ]
    
    # 1. Incident distribution
    print("\n1. Crime Distribution:")
    cells = discretize_point_features(incidents, level=13)
    
    print(f"   Cells with incidents: {len(cells)}")
    for token, data in list(cells.items())[:3]:
        print(f"   Cell {token}: {data['num_points']} incident(s)")
    
    # 2. Hotspot analysis
    print("\n2. Crime Hotspot Analysis:")
    hotspots = discretize_point_clusters(
        incidents,
        level=13,
        cluster_by='cluster_id',
        aggregate_attributes=['severity']
    )
    
    for token, data in list(hotspots.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Hotspots: {list(data['clusters'].keys())}")
        print(f"      Total incidents: {data['total_points']}")
        print(f"      Dominant hotspot: {data.get('dominant_cluster', 'N/A')}")
    
    # 3. Temporal analysis
    print("\n3. Temporal Pattern Analysis:")
    temporal = discretize_point_temporal(
        incidents,
        level=13,
        time_attribute='timestamp'
    )
    
    for token, data in list(temporal.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Total events: {data['total_events']}")
        if 'first_event' in data:
            print(f"      First: {data['first_event']}")
            print(f"      Last: {data['last_event']}")
    
    # 4. Density with kernel smoothing
    print("\n4. Crime Density (Kernel Density Estimation):")
    kde = discretize_point_density(
        incidents,
        level=13,
        normalize_by_area=True,
        kernel_radius_cells=1
    )
    
    high_density = {k: v for k, v in kde.items() if v.get('kernel_count', 0) >= 2}
    print(f"   High-density cells: {len(high_density)}")
    for token, data in list(high_density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Incidents: {data['point_count']}")
        print(f"      Kernel count: {data.get('kernel_count', 0)}")


def example_6_hospitals():
    """Example: Hospital facility locations."""
    print("\n" + "="*70)
    print("Example 6: Hospital Facilities")
    print("="*70)
    
    # Create hospitals
    hospitals = [
        PointFeature(
            'hosp_001', 42.0, -71.0,
            attributes={'beds': 500, 'trauma_level': 1, 'specialties': 5},
            feature_type='hospital',
            weight=2.0  # Major hospital
        ),
        PointFeature(
            'hosp_002', 42.05, -71.05,
            attributes={'beds': 200, 'trauma_level': 3, 'specialties': 3},
            feature_type='hospital',
            weight=1.0
        ),
        PointFeature(
            'hosp_003', 42.1, -71.1,
            attributes={'beds': 350, 'trauma_level': 2, 'specialties': 4},
            feature_type='hospital',
            weight=1.5
        )
    ]
    
    # 1. Hospital coverage
    print("\n1. Hospital Coverage:")
    cells = discretize_point_features(hospitals, level=12)
    
    print(f"   Cells with hospitals: {len(cells)}")
    for token, data in cells.items():
        print(f"   Cell {token}: {data['num_points']} hospital(s)")
    
    # 2. Capacity analysis
    print("\n2. Healthcare Capacity:")
    capacity = discretize_point_attributes(
        hospitals,
        attributes=['beds', 'trauma_level', 'specialties'],
        aggregation_funcs={
            'beds': 'sum',
            'trauma_level': 'min',  # Best trauma level
            'specialties': 'max'
        },
        weight_by=None  # Use feature.weight
    )
    
    for token, data in capacity.items():
        print(f"   Cell {token}:")
        print(f"      Total beds: {data.get('beds_sum', 0):.0f}")
        print(f"      Best trauma level: {data.get('trauma_level_min', 0):.0f}")
        print(f"      Max specialties: {data.get('specialties_max', 0):.0f}")
    
    # 3. Spatial distribution
    print("\n3. Hospital Distribution Pattern:")
    patterns = discretize_point_patterns(
        hospitals,
        level=11,  # Larger cells for regional view
        pattern_metrics=['centroid', 'spread']
    )
    
    for token, data in patterns.items():
        print(f"   Cell {token}:")
        print(f"      Hospitals: {data['num_points']}")
        print(f"      Centroid: ({data.get('centroid', (0, 0))[0]:.3f}, "
              f"{data.get('centroid', (0, 0))[1]:.3f})")
        print(f"      Spread: {data.get('spread_km', 0):.2f} km")


def example_7_tree_inventory():
    """Example: Urban tree inventory."""
    print("\n" + "="*70)
    print("Example 7: Urban Tree Inventory")
    print("="*70)
    
    # Create tree points
    trees = [
        PointFeature(
            'tree_001', 40.8, -73.95,
            attributes={'species': 'oak', 'height_m': 15, 'diameter_cm': 45, 'health': 'good'},
            feature_type='tree'
        ),
        PointFeature(
            'tree_002', 40.801, -73.951,
            attributes={'species': 'maple', 'height_m': 12, 'diameter_cm': 38, 'health': 'fair'},
            feature_type='tree'
        ),
        PointFeature(
            'tree_003', 40.802, -73.949,
            attributes={'species': 'oak', 'height_m': 18, 'diameter_cm': 52, 'health': 'good'},
            feature_type='tree'
        ),
        PointFeature(
            'tree_004', 40.805, -73.955,
            attributes={'species': 'pine', 'height_m': 20, 'diameter_cm': 40, 'health': 'excellent'},
            feature_type='tree'
        ),
        PointFeature(
            'tree_005', 40.803, -73.950,
            attributes={'species': 'maple', 'height_m': 14, 'diameter_cm': 42, 'health': 'good'},
            feature_type='tree'
        )
    ]
    
    # 1. Tree distribution
    print("\n1. Tree Distribution:")
    cells = discretize_point_features(trees, level=14)  # Fine resolution
    
    print(f"   Cells with trees: {len(cells)}")
    for token, data in list(cells.items())[:3]:
        print(f"   Cell {token}: {data['num_points']} tree(s)")
    
    # 2. Canopy metrics
    print("\n2. Canopy Metrics:")
    canopy = discretize_point_attributes(
        trees,
        attributes=['height_m', 'diameter_cm'],
        aggregation_funcs={
            'height_m': 'mean',
            'diameter_cm': 'mean'
        }
    )
    
    for token, data in list(canopy.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Avg height: {data.get('height_m_mean', 0):.1f} m")
        print(f"      Avg diameter: {data.get('diameter_cm_mean', 0):.1f} cm")
    
    # 3. Tree density
    print("\n3. Tree Density:")
    density = discretize_point_density(trees, level=14, normalize_by_area=True)
    
    for token, data in list(density.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      Trees: {data['point_count']}")
        print(f"      Density: {data.get('density_points_per_km2', 0):.1f} trees/km²")
    
    # 4. Spatial clustering
    print("\n4. Tree Clustering (Spatial Proximity):")
    tree_clusters = spatial_cluster_points(
        trees,
        distance_threshold_km=0.5,
        min_cluster_size=2
    )
    
    print(f"   Clusters found: {len(tree_clusters)}")
    for cluster in tree_clusters:
        print(f"   {cluster.cluster_id}: {cluster.num_points} trees")
        print(f"      Centroid: ({cluster.centroid[0]:.4f}, {cluster.centroid[1]:.4f})")


def example_8_air_quality_sensors():
    """Example: Air quality monitoring network."""
    print("\n" + "="*70)
    print("Example 8: Air Quality Sensor Network")
    print("="*70)
    
    # Create air quality sensors
    sensors = [
        PointFeature(
            'aqi_001', 34.05, -118.25,
            attributes={'pm25': 35, 'pm10': 50, 'ozone': 45, 'no2': 30},
            feature_type='air_quality_sensor',
            weight=1.0
        ),
        PointFeature(
            'aqi_002', 34.06, -118.26,
            attributes={'pm25': 42, 'pm10': 58, 'ozone': 52, 'no2': 35},
            feature_type='air_quality_sensor',
            weight=1.0
        ),
        PointFeature(
            'aqi_003', 34.07, -118.24,
            attributes={'pm25': 28, 'pm10': 45, 'ozone': 38, 'no2': 25},
            feature_type='air_quality_sensor',
            weight=1.2  # Higher quality sensor
        ),
        PointFeature(
            'aqi_004', 34.08, -118.27,
            attributes={'pm25': 38, 'pm10': 55, 'ozone': 48, 'no2': 32},
            feature_type='air_quality_sensor',
            weight=1.0
        )
    ]
    
    # 1. Sensor coverage
    print("\n1. Sensor Network Coverage:")
    cells = discretize_point_features(sensors, level=13)
    
    print(f"   Cells monitored: {len(cells)}")
    for token, data in cells.items():
        print(f"   Cell {token}: {data['num_points']} sensor(s)")
    
    # 2. Air quality aggregation
    print("\n2. Air Quality Metrics:")
    aqi_data = discretize_point_attributes(
        sensors,
        attributes=['pm25', 'pm10', 'ozone', 'no2'],
        aggregation_funcs={
            'pm25': 'weighted_mean',
            'pm10': 'max',
            'ozone': 'mean',
            'no2': 'mean'
        }
    )
    
    for token, data in list(aqi_data.items())[:2]:
        print(f"   Cell {token}:")
        print(f"      PM2.5: {data.get('pm25_weighted_mean', 0):.1f} μg/m³")
        print(f"      PM10 (max): {data.get('pm10_max', 0):.0f} μg/m³")
        print(f"      Ozone: {data.get('ozone_mean', 0):.1f} ppb")
        print(f"      NO2: {data.get('no2_mean', 0):.1f} ppb")
    
    # 3. Pollution statistics
    print("\n3. PM2.5 Statistics:")
    pm25_stats = calculate_point_statistics(sensors, 'pm25')
    print(f"   Mean: {pm25_stats['mean']:.1f} μg/m³")
    print(f"   Std Dev: {pm25_stats['std']:.1f} μg/m³")
    print(f"   Range: {pm25_stats['min']:.0f} - {pm25_stats['max']:.0f} μg/m³")
    
    # 4. Filter high pollution areas
    print("\n4. High Pollution Sensors:")
    high_pollution = filter_points_by_attribute(
        sensors,
        'pm25',
        lambda x: x > 35
    )
    print(f"   Sensors with PM2.5 > 35: {len(high_pollution)}")
    for sensor in high_pollution:
        print(f"      {sensor.feature_id}: {sensor.get_attribute('pm25')} μg/m³")


def example_9_dict_based_input():
    """Example: Creating points from dictionary (e.g., JSON, database)."""
    print("\n" + "="*70)
    print("Example 9: Dictionary-based Point Creation")
    print("="*70)
    
    # Point data from JSON/database
    point_data = {
        'feature_id': 'poi_12345',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'attributes': {
            'name': 'Golden Gate Park',
            'type': 'park',
            'area_acres': 1017,
            'visitors_annual': 13000000
        },
        'feature_type': 'poi',
        'weight': 2.0
    }
    
    # Create feature from dict
    point = create_point_feature_from_dict(point_data)
    
    print(f"\n1. Created point from dictionary:")
    print(f"   Feature ID: {point.feature_id}")
    print(f"   Location: ({point.latitude}, {point.longitude})")
    print(f"   Type: {point.feature_type}")
    print(f"   Attributes: {point.attributes}")
    
    # Discretize
    cells = discretize_point_features([point], level=13)
    
    print(f"\n2. Discretization result:")
    for token, data in cells.items():
        print(f"   Cell: {token}")
        print(f"   Features: {data['features']}")
    
    # Export to dict
    export_data = point.to_dict()
    
    print(f"\n3. Export data:")
    print(json.dumps(export_data, indent=2))


def run_all_examples():
    """Run all point discretization examples."""
    print("\n" + "#"*70)
    print("# POINT DISCRETIZATION EXAMPLES")
    print("# Generic framework for point-based spatial data")
    print("#"*70)
    
    example_1_weather_stations()
    example_2_points_of_interest()
    example_3_soil_sampling()
    example_4_cell_towers()
    example_5_crime_incidents()
    example_6_hospitals()
    example_7_tree_inventory()
    example_8_air_quality_sensors()
    example_9_dict_based_input()
    
    print("\n" + "#"*70)
    print("# All examples completed successfully!")
    print("#"*70 + "\n")


if __name__ == '__main__':
    run_all_examples()
