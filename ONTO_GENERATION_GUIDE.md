# Ontology/Schema Generation Guide

## Overview

`onto_generation` 模块提供**两种方法**自动从输入数据中生成本体/Schema：

### 方法对比

| 方法 | 输入 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| **自上而下** | 能力问题 (Competency Questions) | 需求明确的系统 | 符合需求，完整性好 | 需要提前定义问题 |
| **自下而上** | 现有的 Triples 数据 | 探索性数据分析 | 从真实数据推导，客观性好 | 可能遗漏高层概念 |

---

## 方法 1：自上而下 - 从能力问题提取本体

### 概念

**能力问题（Competency Questions）**：系统应该能够回答的问题，反映了本体应该覆盖的知识范围。

```python
from GraphConstruct import generate_ontology_from_questions

# 定义能力问题
questions = [
    "Which persons work for which organizations?",
    "What are the events participated by persons?",
    "Which organizations are located in which locations?",
    "What concepts are related to events?",
    "Which projects are managed by persons?"
]

# 生成本体
ontology = generate_ontology_from_questions(questions, verbatim=True)
```

### 使用 TopDownOntologyExtractor

```python
from GraphConstruct import TopDownOntologyExtractor

extractor = TopDownOntologyExtractor()

questions = [
    "Which persons work for which organizations?",
    "What events did persons participate in?",
    "Which locations hosted events?"
]

# 提取本体
ontology = extractor.extract_from_competency_questions(
    questions=questions,
    verbatim=True
)

# 查看结果
print("Entity Types:")
for entity_type, info in ontology["entity_types"].items():
    print(f"  - {entity_type}: {info['properties']}")

print("\nRelation Types:")
for relation_name, info in ontology["relation_types"].items():
    print(f"  - {relation_name}: {info['domain']} → {info['range']}")
```

**输出示例**：
```
Entity Types:
  - Person: ['name', 'age', 'role', 'email']
  - Organization: ['name', 'type', 'founded_date', 'location']
  - Event: ['name', 'date', 'location', 'description']
  - Location: ['name', 'type', 'coordinates']

Relation Types:
  - works_for: Person → Organization
  - participated_in: Person → Event
  - located_in: Organization → Location
```

---

## 方法 2：自下而上 - 从 Triples 归纳本体

### 概念

从现有的 triples 数据中自动分析和归纳实体类型和关系类型。

```python
from GraphConstruct import generate_ontology_from_triples

# 假设你已经有了 triples 数据
triples = [
    {"node_1": "Alice", "node_1_type": "Person", "edge": "works_for", "node_2": "MIT", "node_2_type": "Organization"},
    {"node_1": "Bob", "node_1_type": "Person", "edge": "works_for", "node_2": "Harvard", "node_2_type": "Organization"},
    {"node_1": "MIT", "node_1_type": "Organization", "edge": "located_in", "node_2": "Boston", "node_2_type": "Location"},
    {"node_1": "Alice", "node_1_type": "Person", "edge": "participated_in", "node_2": "Conference2024", "node_2_type": "Event"},
]

# 从 triples 生成本体
ontology = generate_ontology_from_triples(
    triples=triples,
    min_frequency=1,
    verbatim=True
)
```

### 使用 BottomUpOntologyInducer

```python
from GraphConstruct import BottomUpOntologyInducer

inducer = BottomUpOntologyInducer()

# 从 triples 归纳本体
ontology = inducer.induce_ontology_from_triples(
    triples=triples,
    min_frequency=2,  # 只保留至少出现2次的实体/关系
    verbatim=True
)

# 查看统计信息
stats = ontology["statistics"]
print(f"Total entities: {stats['total_entities']}")
print(f"Total relations: {stats['total_relations']}")
print(f"Total triples: {stats['total_triples']}")
```

### TripleAnalyzer 详细分析

```python
from GraphConstruct import TripleAnalyzer

analyzer = TripleAnalyzer()

# 分析 triples
analysis = analyzer.analyze_triples(triples)

# 查看分布
print("Entity distribution:")
for entity, count in analysis['entity_distribution'].items():
    print(f"  {entity}: {count}")

print("\nRelation distribution:")
for relation, count in analysis['relation_distribution'].items():
    print(f"  {relation}: {count}")

# 推导实体类型
entity_types = analyzer.infer_entity_types(triples, verbatim=True)

# 推导关系类型
relation_types = analyzer.infer_relation_types(triples, verbatim=True)
```

---

## 完整工作流程示例

### 场景 1：使用能力问题定义系统要求

```python
from GraphConstruct import (
    generate_ontology_from_questions,
    ontology_to_graphschema,
    make_graph_from_text
)

# 第 1 步：定义能力问题
competency_questions = [
    "Which scientists contributed to which publications?",
    "What research areas did scientists work in?",
    "Which publications cite which other publications?",
    "Where are research institutions located?"
]

# 第 2 步：从问题生成本体
ontology = generate_ontology_from_questions(
    competency_questions,
    verbatim=True
)

# 第 3 步：转换为 GraphSchema 用于验证
schema = ontology_to_graphschema(ontology)

# 第 4 步：使用 schema 生成图
text = """
Dr. Alice published a paper on Machine Learning at MIT.
Bob published a paper on Natural Language Processing at Stanford.
The NLP paper cites the ML paper.
"""

graph_HTML, graph_GraphML, G, net, output_pdf, validation_stats = make_graph_from_text(
    text,
    generate_fn=your_llm_function,
    schema=schema,
    validate_against_schema=True,
    normalize_entities=True
)

print(f"Validated triples: {validation_stats['after_validation']}")
```

### 场景 2：从现有数据探索本体

```python
from GraphConstruct import (
    generate_ontology_from_triples,
    make_graph_from_text
)

# 第 1 步：从之前的图生成中提取 triples
# (假设你已经有了 CSV 或其他格式的 triples)
import pandas as pd

triples_df = pd.read_csv('extracted_triples.csv')
triples = triples_df.to_dict('records')

# 第 2 步：从数据归纳本体
inferred_ontology = generate_ontology_from_triples(
    triples,
    min_frequency=3,  # 只保留频繁出现的类型
    verbatim=True
)

# 第 3 步：审查自动生成的本体
print("Discovered entity types:")
for ent_type, info in inferred_ontology['entity_types'].items():
    print(f"  {ent_type}: {len(info['examples'])} instances")

print("\nDiscovered relations:")
for rel_name, info in inferred_ontology['relation_types'].items():
    print(f"  {rel_name}: {info['frequency']} occurrences")
```

### 场景 3：合并多个本体

```python
from GraphConstruct import OntologyMerger

# 假设你从不同来源生成了多个本体
onto1 = generate_ontology_from_questions(questions_set_1)
onto2 = generate_ontology_from_triples(triples_set_1)
onto3 = generate_ontology_from_triples(triples_set_2)

# 合并本体
# 方式 1：并集 - 保留所有发现的类型
merged_union = OntologyMerger.merge_ontologies(
    [onto1, onto2, onto3],
    strategy="union",
    verbatim=True
)

# 方式 2：交集 - 只保留所有来源都有的类型
merged_intersection = OntologyMerger.merge_ontologies(
    [onto1, onto2, onto3],
    strategy="intersection",
    verbatim=True
)
```

---

## API 参考

### TopDownOntologyExtractor

```python
class TopDownOntologyExtractor:
    def extract_from_competency_questions(
        questions: List[str],
        verbatim: bool = False
    ) -> Dict
```

**参数**：
- `questions`: 能力问题列表
- `verbatim`: 是否打印详细信息

**返回**：包含 `entity_types` 和 `relation_types` 的字典

### BottomUpOntologyInducer

```python
class BottomUpOntologyInducer:
    def induce_ontology_from_triples(
        triples: List[Dict],
        min_frequency: int = 2,
        verbatim: bool = False
    ) -> Dict
```

**参数**：
- `triples`: triple 字典列表，每个应包含: `node_1`, `node_1_type`, `edge`, `node_2`, `node_2_type`
- `min_frequency`: 最小出现次数阈值
- `verbatim`: 是否打印详细信息

**返回**：包含推导的实体类型、关系类型和统计信息的字典

### TripleAnalyzer

```python
class TripleAnalyzer:
    def analyze_triples(triples: List[Dict]) -> Dict
    def infer_entity_types(
        triples: List[Dict],
        method: EntityTypeInferenceMethod = EntityTypeInferenceMethod.PATTERN,
        min_frequency: int = 2,
        verbatim: bool = False
    ) -> Dict[str, InferredEntityType]
    def infer_relation_types(
        triples: List[Dict],
        min_frequency: int = 2,
        verbatim: bool = False
    ) -> Dict[str, InferredRelationType]
```

### 便利函数

```python
# 从能力问题生成本体
def generate_ontology_from_questions(
    questions: List[str],
    generate_fn = None,
    verbatim: bool = False
) -> Dict

# 从 triples 生成本体
def generate_ontology_from_triples(
    triples: List[Dict],
    min_frequency: int = 2,
    verbatim: bool = False
) -> Dict

# 将本体转换为 GraphSchema
def ontology_to_graphschema(ontology: Dict) -> GraphSchema
```

---

## 最佳实践

### 1. 结合两种方法

**推荐工作流**：

1. 从能力问题定义**初始本体** (自上而下)
2. 从真实数据验证和**扩展本体** (自下而上)
3. 使用**合并工具**协调两个本体

```python
# 初始本体（需求驱动）
initial_onto = generate_ontology_from_questions(
    competency_questions
)

# 发现性本体（数据驱动）
discovered_onto = generate_ontology_from_triples(
    existing_triples,
    min_frequency=3
)

# 合并得到完整本体
final_onto = OntologyMerger.merge_ontologies(
    [initial_onto, discovered_onto],
    strategy="union"
)
```

### 2. 调整最小频率阈值

- **探索性分析**：使用 `min_frequency=1`，发现所有类型
- **生产系统**：使用 `min_frequency=3`，只保留常见类型
- **质量控制**：使用 `min_frequency=5`，严格过滤

### 3. 审查推导结果

总是检查推导出的本体是否符合常识：

```python
ontology = generate_ontology_from_triples(triples, verbatim=True)

# 手工审查每个关系的 domain/range
for rel_name, rel_info in ontology['relation_types'].items():
    domain = rel_info['domain']
    range_ = rel_info['range']
    
    # 验证是否合理
    if not is_sensible_domain_range(domain, range_):
        print(f"⚠️ Warning: {rel_name} ({domain}→{range_}) seems odd")
```

### 4. 版本控制本体

保存生成的本体为 JSON，便于版本管理和比对：

```python
import json

ontology = generate_ontology_from_triples(triples)

# 保存
with open('ontology_v1.json', 'w') as f:
    json.dump(ontology, f, indent=2, default=str)

# 加载
with open('ontology_v1.json', 'r') as f:
    ontology = json.load(f)

# 转为 GraphSchema
schema = ontology_to_graphschema(ontology)
```

### 5. 集成到图生成流程

```python
from GraphConstruct import (
    make_graph_from_text,
    generate_ontology_from_questions,
    ontology_to_graphschema
)

# 第 1 步：生成本体
ontology = generate_ontology_from_questions(
    questions=['What does X do?', 'Where is X located?']
)

# 第 2 步：转为 Schema
schema = ontology_to_graphschema(ontology)

# 第 3 步：使用 Schema 生成图
for text in text_collection:
    graph_HTML, graph_GraphML, G, net, output_pdf, val_stats = make_graph_from_text(
        text,
        generate_fn=llm_fn,
        schema=schema,
        validate_against_schema=True
    )
    
    # 监控质量
    if val_stats['validation_failure_rate'] > 0.3:
        print(f"⚠️ High failure rate: {val_stats['validation_failure_rate']}")
```

---

## 故障排除

### Q: 推导出的实体类型太多/太少

**A**: 调整 `min_frequency` 参数
```python
# 更宽松 - 发现更多类型
ontology = generate_ontology_from_triples(triples, min_frequency=1)

# 更严格 - 只保留常见类型
ontology = generate_ontology_from_triples(triples, min_frequency=5)
```

### Q: 关系的 domain/range 推导不对

**A**: 通常是因为数据噪声。使用高于门槛的 `min_frequency` 或手工调整：

```python
# 手工修正
ontology['relation_types']['works_for']['domain'] = 'Person'
ontology['relation_types']['works_for']['range'] = 'Organization'
```

### Q: 能力问题提取的本体太简单

**A**: 添加更多具体的问题：

```python
# 不够具体
questions = ["What are people?", "What are organizations?"]

# 更具体
questions = [
    "Which scientists published which papers?",
    "Which papers cite which other papers?",
    "At which institutions do scientists work?",
    "What research areas do papers address?"
]
```

---

## 输出格式说明

### 本体字典结构

```python
{
    "entity_types": {
        "Person": {
            "name": "Person",
            "frequency": 42,
            "examples": ["Alice", "Bob", "Charlie"],
            "description": "...",
            "properties": ["name", "age", "email"],
            "confidence": 0.95
        },
        ...
    },
    "relation_types": {
        "works_for": {
            "name": "works_for",
            "domain": "Person",
            "range": "Organization",
            "frequency": 23,
            "examples": [("Alice", "MIT"), ("Bob", "Harvard")],
            "confidence": 0.98
        },
        ...
    },
    "statistics": {
        "total_entities": 156,
        "total_relations": 42,
        "total_triples": 1024
    }
}
```

---

## 总结

| 特性 | 自上而下 | 自下而上 |
|------|---------|---------|
| **适用场景** | 需求明确、系统设计初期 | 数据驱动、本体发现 |
| **输入** | 能力问题 | 现有 triples |
| **输出** | 完整本体框架 | 数据驱动的本体 |
| **优点** | 符合要求、可控性高 | 客观、基于实际数据 |
| **缺点** | 需提前设计、可能遗漏 | 可能有噪声、不完整 |
| **函数** | `generate_ontology_from_questions()` | `generate_ontology_from_triples()` |

**推荐**：结合两种方法使用，自上而下定义框架，自下而上验证和扩展。
