# LLM-Based Bottom-Up 本体生成 - 快速参考

## 概述

基于论文 "Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation" 实现的 LLM-based Bottom-Up 本体生成方法。

从现有的知识图谱三元组出发，使用 LLM 自动生成本体/模式。

## 三种提取策略

| 策略 | 名称 | 特点 | 最佳用途 |
|------|------|------|--------|
| **instance** | 实例级 | 从单个实例提取类型 | 明确的实例分类 |
| **pattern** | 模式级 | 识别多实例间的模式 | 发现关系模式 |
| **semantic** | 语义级 | 深层语义理解 | 综合分析，高质量 |

## 快速开始

### 基本导入

```python
from GraphConstruct import (
    # Generator class
    LLMBasedBottomUpGenerator,
    
    # Convenience functions
    generate_ontology_llm_bottomup,
    compare_bottomup_methods,
)

from Llms.llm_providers import get_generate_fn
```

### 基础用法

```python
from GraphConstruct import generate_ontology_llm_bottomup

# 准备三元组
triples = [
    {
        "node_1": "Alice",
        "node_2": "Bob",
        "edge": "knows",
        "node_1_type": "Person",
        "node_2_type": "Person"
    },
    {
        "node_1": "Bob",
        "node_2": "MIT",
        "edge": "works_for",
        "node_1_type": "Person",
        "node_2_type": "Organization"
    }
]

# 生成本体 - 使用语义策略
ontology = generate_ontology_llm_bottomup(
    triples,
    strategy='semantic',
    verbatim=True
)

print(f"Entity types: {len(ontology['entity_types'])}")
print(f"Relation types: {len(ontology['relation_types'])}")
```

## 详细指南

### 策略 1: 实例级 (Instance-Level)

**原理**: 从单个实例提取其类型和属性

```python
ontology = generate_ontology_llm_bottomup(
    triples,
    strategy='instance',
    verbatim=True
)
```

**工作流程**:
1. 按类型分组实例
2. 为每种类型选择代表性实例
3. 使用 LLM 验证和增强类型定义
4. 提取每个实例的属性

**适用场景**:
- 实体已明确分类
- 需要详细的实例级信息
- 小到中等规模数据集

### 策略 2: 模式级 (Pattern-Based)

**原理**: 识别多个实例之间的共同模式

```python
ontology = generate_ontology_llm_bottomup(
    triples,
    strategy='pattern',
    verbatim=True
)
```

**工作流程**:
1. 按关系分组三元组
2. 分析领域-范围对
3. 使用 LLM 识别模式
4. 提取关系语义

**适用场景**:
- 关系模式复杂
- 需要理解关系类型
- 发现隐藏的结构

### 策略 3: 语义级 (Semantic-Level)

**原理**: 全局语义理解，综合分析所有三元组

```python
ontology = generate_ontology_llm_bottomup(
    triples,
    strategy='semantic',
    verbatim=True
)
```

**工作流程**:
1. 将三元组转换为自然语言描述
2. 一次性发送给 LLM 进行全面分析
3. LLM 生成完整的本体定义
4. 包括领域/主题识别

**适用场景**:
- 需要高质量本体
- 数据集相对较小（<20 个三元组推荐）
- 需要识别领域
- 寻求最高精度

## 高级用法

### 使用生成器类

对处理有更多控制：

```python
from GraphConstruct import LLMBasedBottomUpGenerator

generator = LLMBasedBottomUpGenerator(
    generate_fn=get_generate_fn("openai", {"model": "gpt-4"}),
    verbatim=True
)

ontology = generator.generate_ontology(
    triples,
    strategy='semantic',
    sample_size=None
)
```

### 处理大型数据集

使用采样处理大型数据集：

```python
# 从 1000 个三元组中采样 20 个
ontology = generate_ontology_llm_bottomup(
    large_triples,
    strategy='semantic',
    sample_size=20,
    verbatim=True
)
```

### 比较所有方法

对比规则基础和 LLM 基础方法：

```python
from GraphConstruct import compare_bottomup_methods

results = compare_bottomup_methods(
    triples,
    generate_fn=generate,
    llm_strategies=['instance', 'pattern', 'semantic'],
    verbatim=True
)

# 访问结果
rule_based_onto = results['rule_based']
instance_onto = results['instance']
pattern_onto = results['pattern']
semantic_onto = results['semantic']

# 查看统计
print(results['comparison_stats'])
```

## 返回值结构

```python
{
    "entity_types": {
        "Person": {
            "name": "Person",
            "definition": "...",
            "properties": ["name", "age"],
            "confidence": 0.95,
            "instances": ["Alice", "Bob"],
            "count": 2
        },
        ...
    },
    "relation_types": {
        "works_for": {
            "name": "works_for",
            "domain": "Person",
            "range": "Organization",
            "semantic_meaning": "...",
            "confidence": 0.85,
            "frequency": 5
        },
        ...
    },
    "metadata": {
        "method": "LLM-based_semantic",  # 或 instance / pattern
        "strategy": "semantic",
        "domain_topic": "Human Resources",  # 仅 semantic 策略
        "total_triples": 10
    }
}
```

## 与规则基础方法的比较

| 特性 | 规则基础 | LLM-based |
|------|---------|----------|
| **精度** | 中等 | 高 |
| **速度** | 快 | 慢 |
| **成本** | 免费 | 需要 API |
| **可解释性** | 高 | 中等 |
| **处理复杂语义** | 否 | 是 |
| **领域理解** | 否 | 是 |

## 实现细节

### Instance-Level 策略

1. 按 `node_1_type` 和 `node_2_type` 分组
2. 为每个类型收集实例
3. LLM 验证: 生成定义、属性、置信度
4. 返回增强的实体类型

### Pattern-Based 策略

1. 按 `edge` 分组三元组
2. 采样领域-范围对
3. LLM 分析: 识别模式、语义含义
4. 返回增强的关系类型

### Semantic-Level 策略

1. 将三元组转换为自然语言
2. 提交给 LLM 进行全面分析
3. LLM 返回完整的本体结构
4. 包括领域识别和语义理解

## 错误处理

所有方法都包含后备机制：

- **LLM 调用失败** → 使用规则基础推理
- **JSON 解析失败** → 返回空字典，使用默认值
- **Semantic 策略失败** → 回退到 Pattern 策略

## 性能考虑

| 策略 | 调用次数 | 时间 | 成本 |
|------|---------|------|------|
| Instance | n（实体数）| 中等 | 中等 |
| Pattern | m（关系数）| 中等 | 中等 |
| Semantic | 1 | 低（相对）| 低（相对）|

其中 n = 实体数，m = 关系数

## 示例场景

### 医学知识图谱

```python
# 医学三元组
medical_triples = [
    {"node_1": "Aspirin", "node_2": "Headache", "edge": "treats", "node_1_type": "Drug", "node_2_type": "Disease"},
    {"node_1": "Aspirin", "node_2": "Fever", "edge": "treats", "node_1_type": "Drug", "node_2_type": "Disease"},
    ...
]

# 语义策略最适合 - 理解医学关系
ontology = generate_ontology_llm_bottomup(
    medical_triples,
    strategy='semantic',
    verbatim=True
)

# 输出: 高质量医学本体，包括药物、疾病、症状等
```

### 电影数据库

```python
# 电影三元组
movie_triples = [
    {"node_1": "Avatar", "node_2": "James Cameron", "edge": "directed_by", ...},
    {"node_1": "Avatar", "node_2": "Science Fiction", "edge": "genre", ...},
    ...
]

# 模式策略最适合 - 识别电影-导演等关系模式
ontology = generate_ontology_llm_bottomup(
    movie_triples,
    strategy='pattern'
)
```

## 与论文的对应关系

本实现直接基于论文的三个方法：

| 本实现 | 论文方法 | 算法 |
|-------|--------|------|
| Instance | Instance-level Extraction | 从实例提取类型 |
| Pattern | Pattern-based Consolidation | 模式识别和合并 |
| Semantic | Semantic-level Refinement | 语义理解精化 |

## 常见问题

**Q: 哪个策略最好？**
A: 这取决于你的数据。推荐使用 `compare_bottomup_methods()` 比较所有方法。

**Q: 需要多少三元组？**
A: 最少 5-10 个，推荐 20-50 个。对于 semantic 策略，建议 <20。

**Q: 可以处理大型数据集吗？**
A: 可以，使用 `sample_size` 参数采样。

**Q: 成本如何？**
A: 每次 LLM 调用都会产生费用。Instance 最便宜，Semantic 最贵但精度最高。

## 参考文献

Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation
