"""
GraphConstruct Comprehensive Documentation
==========================================

A complete guide to using the GraphConstruct package for historical knowledge graph construction.
"""

# Historical Knowledge Graph Construction with GraphConstruct

## Overview

**GraphConstruct** is a comprehensive package for generating knowledge graphs from historical texts and PDF documents. It provides a complete pipeline for:

1. **Document Processing**: Load and parse various document formats (TXT, PDF, Markdown)
2. **Entity Extraction**: Identify persons, places, events, dates, and organizations
3. **Relationship Extraction**: Discover connections between entities
4. **Knowledge Graph Construction**: Build structured knowledge graphs
5. **Graph Export**: Export in multiple formats (JSON, GraphML, CSV, RDF)

## Architecture

```
┌─────────────────┐
│   Documents     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Document Processor      │ ← Load TXT, PDF, Markdown
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Entity Extractor        │ ← Extract entities + types
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Relationship Extractor  │ ← Extract relationships
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ KG Builder              │ ← Build graph structure
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Graph Exporters          │ ← Export (JSON, GraphML, CSV, RDF)
└──────────────────────────┘
```

## Quick Start

### Installation

```python
# Install dependencies
pip install networkx PyPDF2

# Import the main class
from GraphConstruct import HistoricalKGBuilder
```

### Basic Usage

```python
# Create builder
builder = HistoricalKGBuilder()

# Load documents
documents = builder.load_documents('path/to/documents/')

# Extract entities
entities = builder.extract_entities(min_confidence=0.5)

# Extract relationships
relationships = builder.extract_relationships(min_confidence=0.5)

# Build graph
graph = builder.build_graph(name='my_history_kg')

# Export graph
builder.export_graph('my_history_kg', 'output/kg.json', format='json')
```

### Full Pipeline in One Call

```python
builder = HistoricalKGBuilder()
graph = builder.run_pipeline(
    source='path/to/documents/',
    output_dir='output/',
    graph_name='history_kg',
    min_confidence=0.5
)
```

## Key Components

### 1. DocumentProcessor

Handles loading and preprocessing of documents.

**Features:**
- Supports TXT, PDF, and Markdown formats
- Automatic file size validation
- Text preprocessing and normalization
- Document chunking for better processing

**Usage:**
```python
from GraphConstruct import DocumentProcessor

processor = DocumentProcessor(max_size_mb=100)

# Load from directory
documents = processor.load_documents('documents/')

# Load specific file types
documents = processor.load_documents('documents/', file_types=['.txt', '.pdf'])

# Split into chunks
chunks = processor.split_into_chunks(document, chunk_size=500, overlap=50)

# Preprocess text
clean_text = processor.preprocess_text(raw_text)
```

### 2. EntityExtractor

Extracts named entities from text.

**Supported Entity Types:**
- PERSON: Individuals, historical figures
- PLACE: Cities, countries, geographic locations
- EVENT: Historical events, battles, wars
- DATE: Years, months, specific dates
- ORGANIZATION: Governments, institutions
- DOCUMENT: Historical documents
- CONCEPT: Abstract ideas and concepts
- ARTIFACT: Physical objects, artworks

**Usage:**
```python
from GraphConstruct import EntityExtractor, EntityType

extractor = EntityExtractor(use_llm=False)

# Extract all entity types
entities = extractor.extract_entities(text)

# Extract specific types
entities = extractor.extract_entities(
    text,
    entity_types=[EntityType.PERSON, EntityType.DATE]
)

# Filter by confidence
filtered = extractor.filter_entities_by_confidence(entities, min_confidence=0.7)

# Deduplicate
unique = extractor.deduplicate_entities(entities)
```

### 3. RelationshipExtractor

Extracts relationships between entities.

**Supported Relationship Types:**

#### Temporal
- BEFORE, AFTER, DURING, CONTEMPORARY_WITH

#### Social
- MARRIED_TO, SIBLING_OF, COLLEAGUE_OF
- RIVAL_OF, ALLY_OF, ENEMY_OF

#### Organizational
- MEMBER_OF, LEADER_OF, FOUNDED, PART_OF

#### Locational
- LOCATED_IN, BORDERS, NEAR, CONTAINS

#### Event-Based
- PARTICIPATED_IN, CAUSED, RESULTED_IN, OCCURRED_IN

#### General
- RELATED_TO, INFLUENCED, OPPOSED, SUPPORTED

**Usage:**
```python
from GraphConstruct import RelationshipExtractor

extractor = RelationshipExtractor(use_llm=False)

# Extract relationships
relationships = extractor.extract_relationships(text, entities)

# Filter by confidence
filtered = extractor.filter_relationships_by_confidence(relationships, min_confidence=0.7)

# Deduplicate
unique = extractor.deduplicate_relationships(relationships)
```

### 4. KnowledgeGraphBuilder

Constructs and exports knowledge graphs.

**Export Formats:**
- JSON: Full structured format
- GraphML: Network visualization format
- CSV: Tabular node and edge files
- RDF: Semantic web format

**Usage:**
```python
from GraphConstruct import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder()

# Create graph
graph = builder.create_graph('my_kg', entities, relationships)

# Export to different formats
builder.export_to_json('my_kg', 'output/kg.json')
builder.export_to_graphml('my_kg', 'output/kg.graphml')
builder.export_to_csv('my_kg', 'output/')
builder.export_to_rdf('my_kg', 'output/kg.ttl')

# Get statistics
stats = graph.get_statistics()
print(f"Nodes: {stats['num_nodes']}")
print(f"Edges: {stats['num_edges']}")
```

### 5. HistoricalKGBuilder

Main orchestrator class that combines all components.

**Usage:**
```python
from GraphConstruct import HistoricalKGBuilder

# Initialize
builder = HistoricalKGBuilder(use_llm=False, llm_provider='openai')

# Step 1: Load documents
documents = builder.load_documents('documents/')

# Step 2: Extract entities
entities = builder.extract_entities(
    min_confidence=0.5,
    entity_types=None  # None = all types
)

# Step 3: Extract relationships
relationships = builder.extract_relationships(min_confidence=0.5)

# Step 4: Build graph
graph = builder.build_graph(name='history_kg')

# Step 5: Export
builder.export_graph('history_kg', 'output/kg.json', format='json')

# Get statistics
stats = builder.get_statistics()
```

## Examples

### Example 1: Simple Text Processing

```python
from GraphConstruct import HistoricalKGBuilder

builder = HistoricalKGBuilder()

# Load a single text file
documents = builder.load_documents('history.txt')

# Extract entities and relationships
entities = builder.extract_entities(min_confidence=0.5)
relationships = builder.extract_relationships(min_confidence=0.5)

# Display results
print(f"Found {len(entities)} entities")
for entity in entities[:5]:
    print(f"  - {entity.name} ({entity.entity_type.value})")

print(f"\nFound {len(relationships)} relationships")
for rel in relationships[:5]:
    print(f"  - {rel.subject.name} --[{rel.predicate.value}]--> {rel.object.name}")
```

### Example 2: Process Multiple Documents

```python
builder = HistoricalKGBuilder()

# Load all documents from directory
documents = builder.load_documents('history_documents/', file_types=['.txt', '.pdf'])

# Process entire pipeline
entities = builder.extract_entities(min_confidence=0.6)
relationships = builder.extract_relationships(min_confidence=0.6)

# Build and export
graph = builder.build_graph(name='multi_doc_kg')
builder.export_graph('multi_doc_kg', 'output/kg.graphml', format='graphml')
```

### Example 3: Entity Type Filtering

```python
from GraphConstruct import EntityType

builder = HistoricalKGBuilder()
builder.load_documents('documents/')

# Extract only persons and dates
entities = builder.extract_entities(
    entity_types=[EntityType.PERSON, EntityType.DATE]
)

# Group by type
by_type = {}
for entity in entities:
    entity_type = entity.entity_type.value
    if entity_type not in by_type:
        by_type[entity_type] = []
    by_type[entity_type].append(entity.name)

for entity_type, names in by_type.items():
    print(f"{entity_type}: {', '.join(names)}")
```

### Example 4: Advanced Configuration

```python
# With LLM support (when LLM integration is implemented)
builder = HistoricalKGBuilder(
    use_llm=False,  # Enable for advanced extraction
    llm_provider='openai',  # or 'huggingface'
    max_doc_size_mb=200  # Allow larger documents
)

# Run complete pipeline
graph = builder.run_pipeline(
    source='history_documents/',
    output_dir='output/kg_results/',
    graph_name='historical_kg',
    min_confidence=0.6
)
```

## Data Structures

### Document
```python
@dataclass
class Document:
    content: str                        # Full document text
    source: str                         # File path
    document_type: str                  # 'text', 'pdf', 'markdown'
    metadata: Dict[str, Any]            # File info
    created_at: datetime                # Processing time
```

### Entity
```python
@dataclass
class Entity:
    name: str                           # Entity name
    entity_type: EntityType             # Type enum
    description: Optional[str]          # Description
    context: Optional[str]              # Context from document
    source_document: Optional[str]      # Source file
    start_position: int                 # Text position
    end_position: int                   # Text position
    confidence: float                   # 0.0-1.0 confidence score
    attributes: Dict[str, Any]          # Custom attributes
```

### Relationship
```python
@dataclass
class Relationship:
    subject: Entity                     # Source entity
    predicate: RelationType             # Relationship type
    object: Entity                      # Target entity
    context: Optional[str]              # Context from document
    confidence: float                   # 0.0-1.0 confidence score
    source_document: Optional[str]      # Source file
    supporting_text: Optional[str]      # Supporting quote
```

### Node & Edge
```python
@dataclass
class Node:
    id: str                             # Unique ID
    label: str                          # Display name
    entity_type: str                    # Entity type
    attributes: Dict[str, Any]          # Properties

@dataclass
class Edge:
    source: str                         # Source node ID
    target: str                         # Target node ID
    label: str                          # Edge label
    relation_type: str                  # Relationship type
    attributes: Dict[str, Any]          # Properties
```

## Configuration

### Entity Extraction Settings

```python
# Extract with confidence threshold
entities = builder.extract_entities(min_confidence=0.5)

# Extract specific types
entities = builder.extract_entities(
    entity_types=[EntityType.PERSON, EntityType.PLACE]
)

# Both combined
entities = builder.extract_entities(
    min_confidence=0.7,
    entity_types=[EntityType.PERSON, EntityType.DATE, EntityType.PLACE]
)
```

### Relationship Extraction Settings

```python
# Extract relationships with confidence threshold
relationships = builder.extract_relationships(min_confidence=0.5)

# Default extraction is pattern-based
# LLM-based extraction (when implemented):
builder = HistoricalKGBuilder(use_llm=True, llm_provider='openai')
relationships = builder.extract_relationships()
```

### Graph Export Settings

```python
# JSON (verbose format)
builder.export_graph('kg', 'output.json', format='json')

# GraphML (for visualization)
builder.export_graph('kg', 'output.graphml', format='graphml')

# CSV (tabular format)
builder.export_graph('kg', 'output/', format='csv')
# Creates: output/kg_nodes.csv, output/kg_edges.csv

# RDF (semantic web format)
builder.export_graph('kg', 'output.ttl', format='rdf')
```

## Integration with DGGS

GraphConstruct can be integrated with DGGS for spatial grounding of historical events:

```python
from GraphConstruct import HistoricalKGBuilder
from Dggs import DggsS2

builder = HistoricalKGBuilder()
graph = builder.run_pipeline('documents/')

# Ground entities in spatial grid
grid = DGGSS2(level=12)

for node in graph.nodes.values():
    if node.entity_type == 'place':
        # Example coordinates (in real use, these would come from entity attributes)
        lat, lon = node.attributes.get('latitude'), node.attributes.get('longitude')
        if lat and lon:
            cell_token = grid.latlon_to_token(lat, lon, 12)
            node.attributes['dggs_token'] = cell_token
```

## Supported Document Formats

### Text Files (.txt)
- Plain text UTF-8 encoded
- Maximum size: 100 MB (configurable)

### PDF Files (.pdf)
- Standard PDF documents
- Text extraction via PyPDF2
- Requires: `pip install PyPDF2`

### Markdown Files (.md, .markdown)
- Markdown formatted text
- Preserves structure during processing

## Performance Considerations

### Document Size
- Default max: 100 MB per document
- Configurable via `max_doc_size_mb` parameter
- Larger documents automatically chunked for processing

### Processing Time
- Depends on document size and content complexity
- Pattern-based extraction: Very fast (milliseconds to seconds)
- LLM-based extraction: Slower but more accurate (requires API calls)

### Memory Usage
- Text documents: ~1-2x document size
- PDF documents: ~2-3x file size during processing
- Graph structure: Proportional to entity/relationship count

## Troubleshooting

### PDF Reading Fails
```
Error: PyPDF2 not installed
Solution: pip install PyPDF2
```

### No Entities Extracted
- Ensure document contains proper names/dates
- Lower confidence threshold: `min_confidence=0.3`
- Check document format and encoding (should be UTF-8)

### Too Many Duplicate Relationships
- Increase confidence threshold
- Use `deduplicate_relationships()` method
- Configure relationship patterns

### Graph Export Fails
```
Error: Graph '{name}' not found
Solution: Ensure graph was built: builder.build_graph(name='{name}')
```

## Future Enhancements

- [ ] LLM-based advanced entity linking
- [ ] Temporal relationship inference
- [ ] Spatial grounding with coordinates
- [ ] Multi-lingual support
- [ ] Interactive web visualization
- [ ] Graph reasoning and inference
- [ ] Confidence-based edge weighting
- [ ] Named entity disambiguation

## Contributing

Contributions welcome! Areas for improvement:
- Additional entity types and relationship types
- LLM integration for advanced extraction
- Performance optimization
- Additional export formats
- Visualization tools

## References

- S2 Hierarchical Discrete Global Grid: https://s2geometry.io/
- RDF/Turtle Format: https://www.w3.org/TR/turtle/
- GraphML Format: http://graphml.graphdrawing.org/
- Knowledge Graph Construction: https://en.wikipedia.org/wiki/Knowledge_graph

## License

Same as GraphReasoning project - See LICENSE file

---

**Version**: 0.1.0  
**Last Updated**: 2024  
**Author**: MIT GraphReasoning Team
