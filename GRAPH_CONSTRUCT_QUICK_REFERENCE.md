"""
GraphConstruct Quick Reference
==============================

Fast lookup guide for common tasks.
"""

# GraphConstruct Quick Reference

## Installation

```bash
# Basic installation
pip install networkx PyPDF2

# In your code
from GraphConstruct import HistoricalKGBuilder, EntityType
```

## Basic Pipeline (One-Liner)

```python
from GraphConstruct import HistoricalKGBuilder

builder = HistoricalKGBuilder()
graph = builder.run_pipeline('documents/', output_dir='output/', graph_name='kg')
```

## Step-by-Step Pipeline

```python
builder = HistoricalKGBuilder()
builder.load_documents('documents/')
builder.extract_entities(min_confidence=0.5)
builder.extract_relationships(min_confidence=0.5)
graph = builder.build_graph(name='my_kg')
builder.export_graph('my_kg', 'output/kg.json', format='json')
```

## Common Tasks

### Load Documents
```python
# Single file
docs = builder.load_documents('file.txt')

# Directory
docs = builder.load_documents('documents/')

# Specific formats
docs = builder.load_documents('docs/', file_types=['.txt', '.pdf'])
```

### Extract Entities
```python
# All types
entities = builder.extract_entities()

# Specific types
entities = builder.extract_entities(
    entity_types=[EntityType.PERSON, EntityType.DATE]
)

# With confidence threshold
entities = builder.extract_entities(min_confidence=0.7)
```

### Available Entity Types
```python
EntityType.PERSON          # Individuals, historical figures
EntityType.PLACE           # Cities, countries, locations
EntityType.EVENT           # Historical events, battles
EntityType.DATE            # Years, months, dates
EntityType.ORGANIZATION    # Governments, institutions
EntityType.DOCUMENT        # Historical documents
EntityType.CONCEPT         # Abstract ideas
EntityType.ARTIFACT        # Physical objects, artworks
```

### Extract Relationships
```python
# Extract relationships
rels = builder.extract_relationships()

# With confidence
rels = builder.extract_relationships(min_confidence=0.7)
```

### Common Relationship Types
```python
RelationType.MARRIED_TO        # Social: Marriage
RelationType.PARENT_OF         # Hierarchical: Family
RelationType.LOCATED_IN        # Spatial: Location
RelationType.MEMBER_OF         # Organizational: Membership
RelationType.LEADER_OF         # Organizational: Leadership
RelationType.PARTICIPATED_IN   # Event: Participation
RelationType.CAUSED            # Event: Causation
RelationType.BEFORE            # Temporal: Before
RelationType.AFTER             # Temporal: After
RelationType.RIVAL_OF          # Social: Rivalry
RelationType.ALLY_OF           # Social: Alliance
```

### Build Graph
```python
graph = builder.build_graph(name='kg_name')
```

### Export Formats
```python
# JSON (full structure)
builder.export_graph('kg', 'output.json', format='json')

# GraphML (visualization)
builder.export_graph('kg', 'output.graphml', format='graphml')

# CSV (tabular)
builder.export_graph('kg', 'output_dir/', format='csv')

# RDF (semantic web)
builder.export_graph('kg', 'output.ttl', format='rdf')
```

### Graph Statistics
```python
stats = graph.get_statistics()
print(f"Nodes: {stats['num_nodes']}")
print(f"Edges: {stats['num_edges']}")
print(f"Node types: {stats['node_types']}")
print(f"Edge types: {stats['edge_types']}")
```

### Get Neighbors
```python
neighbors = graph.get_neighbors(node_id)
```

### Process Specific Formats

#### From Text
```python
builder.load_documents('document.txt')
```

#### From PDF (requires PyPDF2)
```python
builder.load_documents('document.pdf')
```

#### From Directory
```python
builder.load_documents('documents_folder/')
```

#### From Markdown
```python
builder.load_documents('notes.md')
```

## Data Access

### Access Extracted Data
```python
# Get entities
entities = builder.entities
for entity in entities:
    print(f"{entity.name}: {entity.entity_type.value}")

# Get relationships
relationships = builder.relationships
for rel in relationships:
    print(f"{rel.subject.name} -> {rel.object.name}")

# Get documents
documents = builder.documents
for doc in documents:
    print(f"Source: {doc.source}")
```

### Access Graph Data
```python
# Get all nodes
nodes = graph.nodes
for node_id, node in nodes.items():
    print(f"{node.label} ({node.entity_type})")

# Get all edges
edges = graph.edges
for edge in edges:
    print(f"{edge.source} -> {edge.target}: {edge.label}")

# Get node neighbors
neighbors = graph.get_neighbors(node_id)
```

## Filtering & Deduplication

### Deduplicate Entities
```python
unique_entities = extractor.deduplicate_entities(entities)
```

### Filter by Confidence
```python
filtered = extractor.filter_entities_by_confidence(entities, min_confidence=0.7)
```

### Deduplicate Relationships
```python
unique_rels = rel_extractor.deduplicate_relationships(relationships)
```

### Filter Relationships by Confidence
```python
filtered_rels = rel_extractor.filter_relationships_by_confidence(rels, min_confidence=0.7)
```

## Advanced Usage

### Custom Entity Processing
```python
from GraphConstruct import EntityExtractor, EntityType

extractor = EntityExtractor()
entities = extractor.extract_entities(text, entity_types=[EntityType.PERSON])

# Get context for entities
for entity in entities:
    print(f"Entity: {entity.name}")
    print(f"Context: {entity.context}")
    print(f"Position: {entity.start_position}-{entity.end_position}")
```

### Custom Relationship Processing
```python
from GraphConstruct import RelationshipExtractor

rel_extractor = RelationshipExtractor()
relationships = rel_extractor.extract_relationships(text, entities)

# Convert to triplets
for rel in relationships:
    subject, predicate, obj = rel.to_triplet()
    print(f"({subject}, {predicate}, {obj})")
```

### Split Large Documents
```python
from GraphConstruct import DocumentProcessor

processor = DocumentProcessor()
chunks = processor.split_into_chunks(
    document,
    chunk_size=500,  # characters
    overlap=50       # overlap between chunks
)
```

### Access Graph Statistics
```python
stats = builder.get_statistics()
print(f"Documents: {stats['documents_loaded']}")
print(f"Entities: {stats['entities_extracted']}")
print(f"Relationships: {stats['relationships_extracted']}")
print(f"Graphs: {stats['graphs']}")
```

## Example: Multi-Document Pipeline

```python
from GraphConstruct import HistoricalKGBuilder, EntityType
from pathlib import Path

builder = HistoricalKGBuilder()

# Process multiple documents
output_dir = Path('output/historical_kg')
output_dir.mkdir(parents=True, exist_ok=True)

# Load and process
builder.load_documents('history_documents/')
builder.extract_entities(min_confidence=0.6)
builder.extract_relationships(min_confidence=0.6)

# Build and export
graph = builder.build_graph(name='historical_kg')
builder.export_graph('historical_kg', str(output_dir / 'kg.json'), format='json')
builder.export_graph('historical_kg', str(output_dir / 'kg.graphml'), format='graphml')

# Print results
stats = builder.get_statistics()
print(f"Built {stats['entities_extracted']} entities")
print(f"Found {stats['relationships_extracted']} relationships")
```

## File Structure

```
GraphConstruct/
├── __init__.py                  # Package init & exports
├── document_processor.py        # Document loading/processing
├── entity_extractor.py          # Entity extraction
├── relationship_extractor.py    # Relationship extraction
├── kg_builder.py                # Knowledge graph construction
└── graph_constructor.py         # Main orchestrator class

examples/
└── graph_construct_examples.py  # Usage examples

Documentation/
├── GRAPH_CONSTRUCT_GUIDE.md              # Full guide (this file)
└── GRAPH_CONSTRUCT_QUICK_REFERENCE.md    # Quick reference
```

## Performance Tips

1. **Use min_confidence threshold** to filter low-confidence extractions
2. **Split large documents** into chunks before processing
3. **Deduplicate early** to reduce processing overhead
4. **Export incrementally** for very large graphs
5. **Use CSV format** for large graphs to avoid memory issues

## Integration with DGGS

```python
from GraphConstruct import HistoricalKGBuilder
from Dggs import DggsS2

builder = HistoricalKGBuilder()
graph = builder.run_pipeline('documents/')

# Spatially ground entities
grid = DGGSS2(level=12)
for node in graph.nodes.values():
    if node.entity_type == 'place':
        # Ground in spatial grid (requires coordinates in attributes)
        lat, lon = node.attributes.get('lat'), node.attributes.get('lon')
        if lat and lon:
            token = grid.latlon_to_token(lat, lon, 12)
            node.attributes['dggs_token'] = token
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `PyPDF2 not installed` | `pip install PyPDF2` |
| No entities extracted | Lower min_confidence or check document format |
| Graph export fails | Ensure graph was built: `builder.build_graph()` |
| Memory issues | Use CSV format, process smaller chunks |
| Duplicate entities | Use `deduplicate_entities()` |
| Too many false relationships | Increase min_confidence threshold |

## See Also

- Full documentation: `GRAPH_CONSTRUCT_GUIDE.md`
- Examples: `examples/graph_construct_examples.py`
- DGGS integration: `DGGS/` package
- GraphReasoning: `GraphReasoning/` package

---

**Last Updated**: 2024  
**Version**: 0.1.0
