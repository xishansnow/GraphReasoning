# Graph Schema and Ontology Usage Guide

## Overview

`make_graph_from_text()` 现在支持 Schema/本体验证，确保提取的知识图谱符合预定义的结构。

## 核心问题解决

### 问题 1: 只提取三元组，不考虑本体
**解决方案**: 使用 `GraphSchema` 定义允许的实体类型和关系类型
```python
schema = GraphSchema(
    entity_types={
        "Person": {"properties": ["name", "role"]},
        "Organization": {"properties": ["name", "type"]},
        "Location": {"properties": ["name", "type"]},
        "Concept": {"properties": ["name", "definition"]},
        "Event": {"properties": ["name", "date"]},
    },
    relation_types={
        "works_for": {"domain": "Person", "range": "Organization"},
        "located_in": {"domain": ["Organization", "Person"], "range": "Location"},
        "participated_in": {"domain": "Person", "range": "Event"},
    }
)
```

### 问题 2: 没有实体类型验证
**解决方案**: 启用 `validate_against_schema=True`
```python
graph_html, graph_graphml, G, net, output_pdf, validation_stats = make_graph_from_text(
    txt=document_text,
    generate=generate_fn,
    schema=schema,
    validate_against_schema=True,  # 启用 Schema 验证
    verbatim=True
)
```

### 问题 3: 同一概念多种表达方式
**解决方案**: 启用 `normalize_entities=True`
```python
graph_html, graph_graphml, G, net, output_pdf, validation_stats = make_graph_from_text(
    txt=document_text,
    generate=generate_fn,
    normalize_entities=True,  # 规范化实体名称
    verbatim=True
)
```

## 使用示例

### 1. 自定义 Schema

```python
from GraphConstruct import GraphSchema, make_graph_from_text

# 定义域特定的本体
medical_schema = GraphSchema(
    entity_types={
        "Disease": {"properties": ["name", "symptoms"]},
        "Symptom": {"properties": ["name", "severity"]},
        "Treatment": {"properties": ["name", "type"]},
        "Drug": {"properties": ["name", "dosage"]},
    },
    relation_types={
        "causes": {"domain": "Disease", "range": "Symptom"},
        "treated_by": {"domain": "Disease", "range": "Treatment"},
        "uses": {"domain": "Treatment", "range": "Drug"},
        "related_to": {"domain": "Disease", "range": "Disease"},
    }
)

# 使用自定义 Schema 生成图
graph_html, graph_graphml, G, net, _, stats = make_graph_from_text(
    txt=medical_text,
    generate=generate_fn,
    schema=medical_schema,
    validate_against_schema=True,
    data_dir="./medical_kg/",
    verbatim=True
)

print(f"验证统计: {stats}")
```

### 2. 验证并过滤三元组

```python
from GraphConstruct import GraphSchema, validate_and_filter_triples

schema = GraphSchema()

triples = [
    {
        "node_1": "Alice",
        "node_2": "Google",
        "edge": "works_for",
        "node_1_type": "Person",
        "node_2_type": "Organization"
    },
    {
        "node_1": "invalid_entity",
        "node_2": "unknown_relation",
        "edge": "unknown_type",
        "node_1_type": "InvalidType",
        "node_2_type": "InvalidType"
    }
]

# 验证并过滤
valid_triples = validate_and_filter_triples(
    triples=triples,
    schema=schema,
    verbatim=True
)

print(f"有效三元组: {len(valid_triples)}")
```

### 3. 规范化实体名称

```python
from GraphConstruct import normalize_entity_names

triples = [
    {"node_1": "John Smith", "node_2": "GOOGLE INC", "edge": "works_for"},
    {"node_1": "john smith", "node_2": "google inc", "edge": "works_for"},  # 重复
    {"node_1": "John Smith", "node_2": "Google Inc", "edge": "works_for"},  # 略有不同
]

# 规范化名称并去重
normalized = normalize_entity_names(triples)
print(f"去重后: {len(normalized)} 个三元组")
```

## 验证统计

函数返回 `validation_stats` 字典，包含：
- `total_extracted`: 提取的总三元组数
- `after_normalization`: 规范化后的三元组数
- `after_validation`: 验证后的三元组数

示例:
```python
validation_stats = {
    "total_extracted": 150,
    "after_normalization": 145,  # 去除了 5 个重复
    "after_validation": 142       # 过滤了 3 个不符合 Schema 的
}
```

## 默认 Schema

如果未提供自定义 Schema，将使用默认本体：

### 实体类型
- `Person`: 人物
- `Organization`: 组织机构
- `Location`: 地理位置
- `Concept`: 概念/主题
- `Event`: 事件

### 关系类型
- `works_for`: Person → Organization
- `located_in`: (Organization|Person|Event) → Location
- `part_of`: Organization → Organization
- `participated_in`: Person → Event
- `related_to`: 通用关系

## 高级功能

### 1. 完整管道

```python
from GraphConstruct import (
    GraphSchema, 
    make_graph_from_text,
    graph_Louvain,
    analyze_network
)

# 步骤 1: 定义 Schema
schema = GraphSchema()

# 步骤 2: 生成图 (带验证)
graph_html, graph_graphml, G, net, output_pdf, stats = make_graph_from_text(
    txt=document_text,
    generate=generate_fn,
    schema=schema,
    validate_against_schema=True,
    normalize_entities=True,
    data_dir="./output/",
    save_HTML=True
)

# 步骤 3: 社区检测
G_communities = graph_Louvain(G, graph_GraphML=graph_graphml)

# 步骤 4: 网络分析
stats = analyze_network(G_communities, data_dir="./output/")

print(f"网络统计: {stats}")
```

### 2. 扩展 Schema

```python
# 创建基础 Schema
base_schema = GraphSchema()

# 扩展为特定领域
extended_schema = GraphSchema(
    entity_types={
        **base_schema.entity_types,  # 保留基础类型
        "Publication": {"properties": ["title", "year"]},
        "Author": {"properties": ["name", "affiliation"]},
    },
    relation_types={
        **base_schema.relation_types,  # 保留基础关系
        "wrote": {"domain": "Author", "range": "Publication"},
        "published_by": {"domain": "Publication", "range": "Organization"},
    }
)
```

## 最佳实践

1. **明确定义本体**: 在提取前清晰定义实体类型和关系类型
2. **启用验证**: 设置 `validate_against_schema=True` 以确保质量
3. **规范化名称**: 使用 `normalize_entities=True` 减少重复
4. **检查统计**: 监控 `validation_stats` 以了解过滤效果
5. **迭代改进**: 根据结果调整 Schema，重新生成更好的图

## 输出文件

生成图时会保存以下文件：
- `{graph_root}_graph.csv`: 原始三元组
- `{graph_root}_graph_clean.csv`: 清理后的三元组
- `{graph_root}_validation_stats.json`: 验证统计
- `{graph_root}_graphML.graphml`: NetworkX 图格式
- `{graph_root}_grapHTML.html`: 可交互的图可视化

## 故障排除

### 问题: 大多数三元组被过滤

**原因**: Schema 定义过严格

**解决方案**: 
1. 检查 `validation_stats.json` 了解被过滤的原因
2. 使用 `verbatim=True` 查看详细的验证信息
3. 扩展 Schema 以包含更多允许的关系

### 问题: 验证后没有三元组

**原因**: LLM 提取的三元组类型与 Schema 不匹配

**解决方案**:
1. 检查 LLM 的 prompt templates 是否与 Schema 一致
2. 考虑添加 LLM 提示来指导实体类型标注
3. 调整 Schema 以适应 LLM 输出

---

**更新日期**: 2025-12-14
**版本**: 1.0
