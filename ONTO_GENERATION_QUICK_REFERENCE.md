# Ontology Generation Quick Reference

## 快速开始

### 方法 1: 从能力问题提取本体（自上而下）

```python
from GraphConstruct import generate_ontology_from_questions

# 定义能力问题
questions = [
    "Which persons work for organizations?",
    "What events did persons participate in?",
    "Which locations host events?"
]

# 生成本体
ontology = generate_ontology_from_questions(questions, verbatim=True)

# 查看结果
for entity_type in ontology["entity_types"]:
    print(f"Entity: {entity_type}")
for rel_type in ontology["relation_types"]:
    print(f"Relation: {rel_type}")
```

### 方法 2: 从 Triples 归纳本体（自下而上）

```python
from GraphConstruct import generate_ontology_from_triples

# 准备 triples 数据
triples = [
    {"node_1": "Alice", "node_1_type": "Person", "edge": "works_for", 
     "node_2": "MIT", "node_2_type": "Organization"},
    {"node_1": "MIT", "node_1_type": "Organization", "edge": "located_in", 
     "node_2": "Boston", "node_2_type": "Location"},
]

# 从 triples 生成本体
ontology = generate_ontology_from_triples(triples, min_frequency=1, verbatim=True)
```

### 方法 3: 转换为 GraphSchema 用于验证

```python
from GraphConstruct import ontology_to_graphschema

# 转换为 schema
schema = ontology_to_graphschema(ontology)

# 用于图生成
from GraphConstruct import make_graph_from_text

graph_HTML, graph_GraphML, G, net, output_pdf, val_stats = make_graph_from_text(
    text,
    generate_fn=llm_fn,
    schema=schema,
    validate_against_schema=True
)
```

---

## 核心类和函数

### TopDownOntologyExtractor

```python
from GraphConstruct import TopDownOntologyExtractor

extractor = TopDownOntologyExtractor()
ontology = extractor.extract_from_competency_questions(
    questions=["Q1?", "Q2?"],
    verbatim=True
)
```

| 方法 | 说明 |
|------|------|
| `extract_from_competency_questions()` | 从能力问题提取本体 |

### BottomUpOntologyInducer

```python
from GraphConstruct import BottomUpOntologyInducer

inducer = BottomUpOntologyInducer()
ontology = inducer.induce_ontology_from_triples(
    triples=triples_list,
    min_frequency=2,
    verbatim=True
)
```

| 方法 | 说明 |
|------|------|
| `induce_ontology_from_triples()` | 从 triples 归纳本体 |

### TripleAnalyzer

```python
from GraphConstruct import TripleAnalyzer

analyzer = TripleAnalyzer()

# 分析 triples
analysis = analyzer.analyze_triples(triples)

# 推导实体类型
entity_types = analyzer.infer_entity_types(triples, min_frequency=2, verbatim=True)

# 推导关系类型
relation_types = analyzer.infer_relation_types(triples, min_frequency=2, verbatim=True)
```

| 方法 | 说明 |
|------|------|
| `analyze_triples()` | 分析 triples 统计信息 |
| `infer_entity_types()` | 推导实体类型 |
| `infer_relation_types()` | 推导关系类型和 domain/range |

### OntologyMerger

```python
from GraphConstruct import OntologyMerger

# 并集：保留所有类型
merged = OntologyMerger.merge_ontologies(
    ontologies=[onto1, onto2, onto3],
    strategy="union"
)

# 交集：只保留共同类型
merged = OntologyMerger.merge_ontologies(
    ontologies=[onto1, onto2, onto3],
    strategy="intersection"
)
```

| 策略 | 说明 |
|------|------|
| `"union"` | 保留所有发现的类型 |
| `"intersection"` | 只保留所有来源都有的类型 |

---

## 便利函数

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

## 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `questions` | 能力问题列表 | - |
| `triples` | Triple 字典列表 | - |
| `min_frequency` | 最小出现次数 | 2 |
| `verbatim` | 打印详细输出 | False |
| `strategy` | 合并策略 (union/intersection) | - |
| `generate_fn` | LLM 生成函数 | None |

---

## Triple 数据格式

Triple 应为字典，包含以下字段：

```python
{
    "node_1": "entity_name",           # 主体名称
    "node_1_type": "EntityType",       # 主体类型
    "edge": "relation_name",           # 关系名
    "node_2": "entity_name",           # 客体名称
    "node_2_type": "EntityType",       # 客体类型
}
```

### 示例

```python
triples = [
    {
        "node_1": "Alice",
        "node_1_type": "Person",
        "edge": "works_for",
        "node_2": "Google",
        "node_2_type": "Organization"
    },
    {
        "node_1": "Google",
        "node_1_type": "Organization",
        "edge": "located_in",
        "node_2": "Mountain View",
        "node_2_type": "Location"
    }
]
```

---

## 输出本体结构

### 本体字典

```python
{
    "entity_types": {
        "Person": {
            "name": "Person",
            "frequency": 42,
            "examples": ["Alice", "Bob"],
            "properties": ["name", "age"],
            "confidence": 0.95
        }
    },
    "relation_types": {
        "works_for": {
            "name": "works_for",
            "domain": "Person",
            "range": "Organization",
            "frequency": 25,
            "examples": [("Alice", "Google")],
            "confidence": 0.98
        }
    },
    "statistics": {
        "total_entities": 156,
        "total_relations": 42,
        "total_triples": 1024
    }
}
```

---

## 完整工作流示例

```python
from GraphConstruct import (
    generate_ontology_from_questions,
    generate_ontology_from_triples,
    OntologyMerger,
    ontology_to_graphschema,
    make_graph_from_text
)

# Step 1: 定义能力问题
questions = [
    "Which persons work for organizations?",
    "What events did persons attend?"
]

# Step 2: 从能力问题生成本体（需求驱动）
onto_requirements = generate_ontology_from_questions(questions)

# Step 3: 从现有数据归纳本体（数据驱动）
onto_discovered = generate_ontology_from_triples(existing_triples)

# Step 4: 合并本体
onto_merged = OntologyMerger.merge_ontologies(
    [onto_requirements, onto_discovered],
    strategy="union"
)

# Step 5: 转为 Schema
schema = ontology_to_graphschema(onto_merged)

# Step 6: 用于图生成
for text in texts:
    graph_HTML, graph_GraphML, G, net, output_pdf, val_stats = make_graph_from_text(
        text,
        generate_fn=llm_fn,
        schema=schema,
        validate_against_schema=True
    )
    print(f"Validated: {val_stats['after_validation']}/{val_stats['total_extracted']}")
```

---

## 调整最小频率

```python
# 发现所有类型（包括低频）
onto_loose = generate_ontology_from_triples(triples, min_frequency=1)

# 平衡模式（推荐）
onto_balanced = generate_ontology_from_triples(triples, min_frequency=2)

# 严格过滤（只保留常见类型）
onto_strict = generate_ontology_from_triples(triples, min_frequency=5)
```

---

## 常见用例

### 用例 1: 探索新数据集

```python
from GraphConstruct import TripleAnalyzer

analyzer = TripleAnalyzer()
analysis = analyzer.analyze_triples(triples)

print(f"Entities: {analysis['unique_entities_count']}")
print(f"Relations: {analysis['unique_relations_count']}")

entity_types = analyzer.infer_entity_types(triples, verbatim=True)
relation_types = analyzer.infer_relation_types(triples, verbatim=True)
```

### 用例 2: 验证数据一致性

```python
onto1 = generate_ontology_from_triples(dataset1)
onto2 = generate_ontology_from_triples(dataset2)

# 找共同部分
common = OntologyMerger.merge_ontologies([onto1, onto2], strategy="intersection")
print(f"Common types: {len(common['entity_types'])} entities, {len(common['relation_types'])} relations")
```

### 用例 3: 系统需求到图生成

```python
# 定义系统需求
requirements = [
    "Track which authors wrote which papers",
    "Record paper citations",
    "Map institutions to locations"
]

# 生成本体
onto = generate_ontology_from_questions(requirements)

# 转为 schema
schema = ontology_to_graphschema(onto)

# 使用 schema 生成图
graph_HTML, _, G, _, _, _ = make_graph_from_text(
    text,
    generate_fn=llm,
    schema=schema,
    validate_against_schema=True
)
```

---

## 保存和加载本体

```python
import json

# 保存
ontology = generate_ontology_from_questions(questions)
with open('my_ontology.json', 'w') as f:
    json.dump(ontology, f, indent=2, default=str)

# 加载
with open('my_ontology.json', 'r') as f:
    onto_loaded = json.load(f)

# 转为 schema
schema = ontology_to_graphschema(onto_loaded)
```

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 推导类型太多 | 增加 `min_frequency` 参数 |
| 推导类型太少 | 减少 `min_frequency` 参数 |
| Domain/range 不对 | 检查输入数据质量，或使用高于门槛的 `min_frequency` |
| 能力问题提取太简单 | 添加更多具体的问题而不是通用问题 |

---

## 相关文档

- [ONTO_GENERATION_GUIDE.md](ONTO_GENERATION_GUIDE.md) - 详细指南
- [GRAPH_SCHEMA_USAGE.md](GRAPH_SCHEMA_USAGE.md) - Schema 使用指南
- [examples/onto_generation_examples.py](examples/onto_generation_examples.py) - 完整示例

---

## 快速命令

```bash
# 查看可用类和函数
python3 -c "from GraphConstruct import *; help(__import__('GraphConstruct.onto_generation', fromlist=['']))"

# 运行示例
python3 examples/onto_generation_examples.py

# 测试导入
python3 -c "from GraphConstruct import generate_ontology_from_questions; print('✓')"
```
