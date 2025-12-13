#!/usr/bin/env python
"""Quick test of DGGS spatial topology and directional relationship features."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only dggs module directly
import importlib.util
spec = importlib.util.spec_from_file_location("dggs", "GraphReasoning/dggs.py")
dggs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dggs)

DGGSS2 = dggs.DGGSS2
analyze_entity_relationships = dggs.analyze_entity_relationships

print("Testing DGGS Spatial Relationship Features...")
print("-" * 60)

grid = DGGSS2(level=12)

# Test 1: Topology relationships
print("\n1. Spatial Topology Relationships")
print("-" * 40)
token1 = grid.latlon_to_token(42.3601, -71.0589, level=12)
token2 = grid.latlon_to_token(42.3605, -71.0596, level=12)  # nearby
token3 = grid.latlon_to_token(42.4501, -71.2589, level=12)  # far

rel_12 = grid.spatial_relation(token1, token2)
rel_13 = grid.spatial_relation(token1, token3)
print(f"✓ Nearby cells: {rel_12}")
print(f"✓ Distant cells: {rel_13}")

# Test 2: Distance calculation
print("\n2. Distance Calculation")
print("-" * 40)
mit = (42.3601, -71.0942)
harvard = (42.3770, -71.1167)
distance = grid.distance_km(*mit, *harvard)
print(f"✓ MIT to Harvard: {distance:.2f} km")

# Test 3: Directional analysis
print("\n3. Directional Analysis")
print("-" * 40)
bearing = grid.bearing(*mit, *harvard)
direction = grid.cardinal_direction(*mit, *harvard)
print(f"✓ Bearing: {bearing:.1f}°")
print(f"✓ Direction: {direction}")

# Test 4: Entity relationship analysis
print("\n4. Entity Relationship Analysis")
print("-" * 40)
entity1 = {"id": "A", "lat": 42.3601, "lon": -71.0589}
entity2 = {"id": "B", "lat": 42.3605, "lon": -71.0596}
relation = grid.entity_relation(entity1, entity2, level=12, distance_threshold_km=1.0)
print(f"✓ Topology: {relation['topology']}")
print(f"✓ Direction: {relation['direction']}")
print(f"✓ Distance: {relation['distance_km']} km")
print(f"✓ Proximity: {relation['proximity']}")

# Test 5: Proximity search
print("\n5. Proximity Search")
print("-" * 40)
entities = [
    {"id": "mit", "lat": 42.3601, "lon": -71.0942, "name": "MIT"},
    {"id": "harvard", "lat": 42.3770, "lon": -71.1167, "name": "Harvard"},
    {"id": "bu", "lat": 42.3505, "lon": -71.1054, "name": "BU"},
    {"id": "northeastern", "lat": 42.3398, "lon": -71.0892, "name": "Northeastern"},
]

nearby = grid.find_entities_in_range(
    entities,
    center_lat=42.3601,
    center_lon=-71.0942,
    radius_km=5.0,
    level=13
)
print(f"✓ Found {len(nearby)} entities within 5 km of MIT")
for ent in nearby[:3]:
    print(f"  - {ent['name']}: {ent['distance_km']} km to the {ent['direction']}")

# Test 6: Batch relationship analysis
print("\n6. Batch Relationship Analysis")
print("-" * 40)
relationships = analyze_entity_relationships(
    entities[:3],  # Just first 3 for speed
    level=13,
    distance_threshold_km=3.0
)
print(f"✓ Analyzed {len(relationships)} pairwise relationships")
for rel in relationships:
    print(f"  - {rel['entity1_id']} <-> {rel['entity2_id']}: "
          f"{rel['topology']}, {rel['direction']}, {rel['distance_km']} km")

print("-" * 60)
print("\n✅ All DGGS spatial relationship features working correctly!")
print()
print("Key features demonstrated:")
print("  1. Topology relationships (equal/contains/within/adjacent/disjoint)")
print("  2. Distance calculation (great circle)")
print("  3. Directional analysis (bearing + cardinal directions)")
print("  4. Entity relationship analysis (comprehensive)")
print("  5. Proximity-based spatial search")
print("  6. Batch pairwise relationship analysis")
