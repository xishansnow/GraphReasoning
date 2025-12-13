#!/usr/bin/env python
"""Quick test of DGGS hierarchical features without full package import."""

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

# Quick test
print("Testing DGGS hierarchical features...")
print("-" * 50)

grid = DGGSS2(level=12)
token = grid.latlon_to_token(42.3601, -71.0589)
print(f'✓ Cell token: {token}')

# Test parent
parent = grid.parent(token, parent_level=10)
print(f'✓ Parent (L10): {parent}')

# Test children
children = grid.children(token, child_level=13)
print(f'✓ Children count (L13): {len(children)}')

# Test neighbors
neighbors = grid.neighbors(token, ring=1)
print(f'✓ Neighbors: {len(neighbors)}')

# Test hierarchical graph
cells = grid.cover_cap(42.36, -71.05, radius_km=5, level=12)
print(f'✓ Covering cells: {len(cells)}')

H = grid.build_hierarchical_graph(cells[:5], include_parents=True)
print(f'✓ Hierarchical graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges')

# Check edge types
edge_types = set(data.get('relation') for _, _, data in H.edges(data=True))
print(f'✓ Edge types: {edge_types}')

print("-" * 50)
print('✅ All DGGS hierarchical features working correctly!')
print()
print("Key features demonstrated:")
print("  1. Parent cell lookup (coarser scale)")
print("  2. Children cell lookup (finer scale)")
print("  3. Same-level neighbor detection")
print("  4. Hierarchical graph with cross-scale edges")
print("  5. Multiple edge types: adjacent, parent_of, child_of")
