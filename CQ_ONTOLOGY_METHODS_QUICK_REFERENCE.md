# CQ-based 本体生成方法快速参考

## 概述

基于论文 "Ontology Generation using Large Language Models" 实现的三种从能力问题（Competency Questions, CQ）生成本体的方法。

## 三种方法对比

| 特性 | CQbyCQ | Memoryless CQbyCQ | Ontogenia |
|------|--------|-------------------|-----------|
| **处理方式** | 迭代处理，保留上下文 | 独立处理，最后合并 | 一次性处理所有 CQ |
| **速度** | 慢（顺序处理） | 中等（可并行） | 快（单次调用） |
| **一致性** | 累积一致性 | 需要合并协调 | 全局一致性 |
| **错误累积** | ⚠️ 可能累积 | ✅ 不累积 | ✅ 不累积 |
| **上下文利用** | ✅ 丰富 | ❌ 无 | ✅ 全局 |
| **适用场景** | 小规模（<20 CQ） | 大规模（>20 CQ） | 中小规模（<15 CQ） |
| **并行化** | ❌ 不支持 | ✅ 支持 | ❌ 不支持 |

---

## 快速开始

### 基本导入

```python
from GraphConstruct import (
    # 生成器类
    CQbyCQGenerator,
    MemorylessCQbyCQGenerator,
    OntogeniaGenerator,
    
    # 便捷函数
    generate_ontology_from_questions,
    compare_cq_methods,
)

from Llms.llm_providers import get_generate_fn
```

### 初始化 LLM

```python
generate = get_generate_fn("openai", config={"model": "gpt-4"})
```

---

## 方法 1：CQbyCQ（迭代式）

### 算法原理

```
初始化空本体
FOR 每个能力问题 CQ:
    呈现当前本体 + 新 CQ 给 LLM
    LLM 提议更新（新类、属性、关系）
    集成更新到本体
返回最终本体
```

### 使用方法

#### 便捷函数

```python
questions = [
    "哪些科学家在哪些机构工作？",
    "科学家参与了哪些研究项目？",
    "研究机构位于哪里？"
]

# 使用 method 参数指定 CQbyCQ 方法
ontology = generate_ontology_from_questions(
    questions=questions,
    generate_fn=generate,
    method='cqbycq',  # ← 指定方法
    verbatim=True     # 显示详细日志
)

print(f"实体类型: {len(ontology['entity_types'])}")
print(f"关系类型: {len(ontology['relation_types'])}")
```

#### 使用生成器类

```python
generator = CQbyCQGenerator(
    generate_fn=generate,
    verbatim=True
)

ontology = generator.generate_ontology(questions)
```

### 优势

✅ **上下文积累**：后续问题利用前面建立的知识  
✅ **迭代改进**：逐步完善本体结构  
✅ **语义一致**：保持累积的语义连贯性  

### 劣势

⚠️ **错误累积**：早期错误可能影响后续处理  
⚠️ **速度较慢**：必须顺序处理  
⚠️ **早期偏差**：可能被早期决策影响  

### 适用场景

- 教育领域知识本体
- 历史事件本体
- 需要深度推理的领域
- CQ 之间有依赖关系

---

## 方法 2：Memoryless CQbyCQ（无记忆式）

### 算法原理

```
FOR 每个 CQ 独立地:
    发送 CQ 给 LLM（无先前上下文）
    LLM 生成本体片段
合并所有片段为单一本体
解决冲突和重复
```

### 使用方法

#### 便捷函数

```python
# 使用 method 参数指定 Memoryless 方法
ontology = generate_ontology_from_questions(
    questions=questions,
    generate_fn=generate,
    method='memoryless',  # ← 指定方法
    verbatim=True
)
```

#### 使用生成器类

```python
generator = MemorylessCQbyCQGenerator(
    generate_fn=generate,
    verbatim=True
)

ontology = generator.generate_ontology(questions)
```

### 优势

✅ **无错误累积**：每个 CQ 独立处理  
✅ **可并行化**：可同时处理多个 CQ  
✅ **鲁棒性强**：单个失败不影响整体  

### 劣势

⚠️ **可能遗漏跨问题关系**：缺少全局视角  
⚠️ **需要复杂合并策略**：解决冲突  
⚠️ **潜在不一致**：片段之间可能矛盾  

### 适用场景

- 调查数据分析
- 多源问题集合
- 大规模 CQ 集（>20 个）
- 需要高吞吐量

---

## 方法 3：Ontogenia（一次性）

### 算法原理

```
一次性呈现所有 CQ 给 LLM
LLM 全局分析所有问题
生成完整、一致的本体（单次传递）
```

### 使用方法

#### 便捷函数

```python
# 使用 method 参数指定 Ontogenia 方法
ontology = generate_ontology_from_questions(
    questions=questions,
    generate_fn=generate,
    method='ontogenia',  # ← 指定方法
    verbatim=True
)
```

#### 使用生成器类

```python
generator = OntogeniaGenerator(
    generate_fn=generate,
    verbatim=True
)

ontology = generator.generate_ontology(questions)
```

### 优势

✅ **全局一致性**：单次生成保证一致  
✅ **识别跨 CQ 模式**：全局视角  
✅ **实现最简单**：单次 LLM 调用  
✅ **速度最快**：无迭代开销  

### 劣势

⚠️ **上下文窗口限制**：CQ 数量受限  
⚠️ **无迭代改进**：一次性结果  
⚠️ **全有或全无**：失败则完全失败  

### 适用场景

- 定义良好的领域
- 快速原型开发
- 小中规模 CQ 集（<15 个）
- 需要快速结果

---

## 方法比较工具

### 同时运行三种方法并比较

```python
results = compare_cq_methods(
    competency_questions=questions,
    generate_fn=generate,
    verbatim=True
)

# 访问各方法结果
cqbycq_onto = results['cqbycq']
memoryless_onto = results['memoryless']
ontogenia_onto = results['ontogenia']

# 统计信息
stats = results['comparison_stats']
print(f"CQbyCQ: {stats['cqbycq_entities']} 实体")
print(f"Memoryless: {stats['memoryless_entities']} 实体")
print(f"Ontogenia: {stats['ontogenia_entities']} 实体")
```

---

## 输出格式

### 本体结构

```python
{
    "entity_types": {
        "Person": {
            "name": "Person",
            "properties": ["name", "age", "email"],
            "description": "表示个人实体"
        },
        "Organization": {
            "name": "Organization",
            "properties": ["name", "type", "location"],
            "description": "表示组织实体"
        }
    },
    "relation_types": {
        "works_for": {
            "name": "works_for",
            "domain": "Person",
            "range": "Organization",
            "description": "人在组织工作"
        }
    },
    "metadata": {
        "method": "CQbyCQ",  # 或 "Memoryless CQbyCQ" 或 "Ontogenia"
        "total_cqs": 5,
        "competency_questions": [...]
    }
}
```

---

## 高级用法

### 1. 保存和加载本体

```python
from GraphConstruct import save_ontology, load_ontology

# 生成本体
ontology = generate_ontology_from_questions(
    questions, 
    generate_fn=generate,
    method='ontogenia'
)

# 保存为不同格式
save_ontology(ontology, "output/ontology.json", format="json")
save_ontology(ontology, "output/ontology.yaml", format="yaml")
save_ontology(ontology, "output/ontology.owl", format="owl")

# 加载
loaded = load_ontology("output/ontology.json")
```

### 2. 转换为 GraphSchema

```python
from GraphConstruct import ontology_to_graphschema

# 将本体转换为 GraphSchema 用于验证
schema = ontology_to_graphschema(ontology)

# 在三元组提取中使用
from GraphConstruct import extract_triples_from_chunks

triples = extract_triples_from_chunks(
    chunks=text_chunks,
    generate=generate,
    schema=schema,
    validate=True  # 使用生成的本体验证
)
```

### 3. 混合方法

```python
# 运行所有三种方法
results = compare_cq_methods(questions, generate)

# 合并结果
combined_entities = {}
for method in ['cqbycq', 'memoryless', 'ontogenia']:
    for entity_name, entity_info in results[method]['entity_types'].items():
        if entity_name not in combined_entities:
            combined_entities[entity_name] = entity_info
        else:
            # 合并属性
            existing = set(combined_entities[entity_name]['properties'])
            new = set(entity_info['properties'])
            combined_entities[entity_name]['properties'] = list(existing | new)

hybrid_ontology = {
    "entity_types": combined_entities,
    "relation_types": {
        # 类似地合并关系
    }
}
```

---

## 性能优化

### 选择合适的 LLM 模型

```python
# 高质量（慢）
generate_gpt4 = get_generate_fn("openai", config={"model": "gpt-4"})

# 平衡（推荐）
generate_gpt35 = get_generate_fn("openai", config={"model": "gpt-3.5-turbo"})

# 本地模型（快，需要硬件）
generate_local = get_generate_fn("local", config={"model": "mixtral:latest"})
```

### 批量处理优化

```python
# 对于大量 CQ，使用 Memoryless 方法
if len(questions) > 20:
    ontology = generate_ontology_from_questions(
        questions, generate, method='memoryless'
    )
# 对于中等数量，使用 Ontogenia
elif len(questions) <= 15:
    ontology = generate_ontology_from_questions(
        questions, generate, method='ontogenia'
    )
# 对于需要深度推理，使用 CQbyCQ
else:
    ontology = generate_ontology_from_questions(
        questions, generate, method='cqbycq'
    )
```

---

## 方法选择指南

### 决策树

```
开始
  │
  ├─ CQ 数量 > 20？
  │    ├─ 是 → 使用 Memoryless CQbyCQ
  │    └─ 否 ↓
  │
  ├─ 需要快速结果？
  │    ├─ 是 → 使用 Ontogenia
  │    └─ 否 ↓
  │
  ├─ CQ 之间有依赖关系？
  │    ├─ 是 → 使用 CQbyCQ
  │    └─ 否 → 使用 Ontogenia
```

### 按场景推荐

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 教育系统本体 | CQbyCQ | 概念之间有层次和依赖 |
| 医疗领域本体 | Ontogenia | 需要全局一致性 |
| 调查数据分析 | Memoryless | 问题独立，可并行 |
| 快速原型 | Ontogenia | 速度优先 |
| 大规模系统 | Memoryless | 可扩展性 |
| 历史知识图谱 | CQbyCQ | 需要时序和因果推理 |

---

## 常见问题

### Q1: 哪种方法最好？

没有"最好"的方法，取决于具体场景：
- **速度优先** → Ontogenia
- **规模优先** → Memoryless CQbyCQ
- **质量优先** → CQbyCQ（小规模时）

### Q2: 可以组合多种方法吗？

可以！运行所有三种方法并合并结果可以获得最全面的本体。

```python
results = compare_cq_methods(questions, generate)
# 然后手动或自动合并结果
```

### Q3: 如何处理冲突的结果？

使用投票或置信度机制：

```python
# 如果某个实体在至少 2 种方法中出现，则保留
entity_votes = defaultdict(int)
for method in ['cqbycq', 'memoryless', 'ontogenia']:
    for entity in results[method]['entity_types']:
        entity_votes[entity] += 1

consensus_entities = {e for e, v in entity_votes.items() if v >= 2}
```

### Q4: 上下文窗口限制怎么办？

对于 Ontogenia 方法，如果 CQ 太多超过上下文：
1. 分批处理 CQ
2. 切换到 Memoryless 方法
3. 使用支持更长上下文的模型

### Q5: 如何评估生成的本体质量？

```python
def evaluate_ontology(ontology, cqs):
    """简单的质量评估指标"""
    metrics = {
        "entity_count": len(ontology['entity_types']),
        "relation_count": len(ontology['relation_types']),
        "avg_properties": np.mean([
            len(e.get('properties', [])) 
            for e in ontology['entity_types'].values()
        ]),
        "coverage": check_cq_coverage(ontology, cqs)
    }
    return metrics
```

---

## 完整工作流示例

```python
from GraphConstruct import (
    generate_ontology_from_questions,
    ontology_to_graphschema,
    extract_triples_from_chunks,
    save_ontology
)

# Step 1: 定义能力问题
questions = [
    "哪些研究人员在哪些项目上工作？",
    "项目由哪些机构资助？",
    "研究成果发表在哪里？"
]

# Step 2: 生成本体
ontology = generate_ontology_from_questions(
    questions,
    generate_fn=generate,
    method='ontogenia',
    verbatim=True
)

# Step 3: 保存本体
save_ontology(ontology, "output/research_ontology.json")

# Step 4: 转换为 Schema
schema = ontology_to_graphschema(ontology)

# Step 5: 使用 Schema 提取三元组
text_chunks = [...]  # 你的文本数据

triples = extract_triples_from_chunks(
    chunks=text_chunks,
    generate=generate,
    schema=schema,
    validate=True
)

# Step 6: 构建知识图谱
from GraphConstruct import make_graph_from_text

graph = make_graph_from_text(
    txt=full_text,
    generate=generate,
    schema=schema,
    validate_against_schema=True
)
```

---

## 总结

三种方法各有优势：

| 方法 | 最佳用途 | 关键优势 |
|------|----------|----------|
| **CQbyCQ** | 深度推理 | 上下文积累 |
| **Memoryless** | 大规模处理 | 可并行化 |
| **Ontogenia** | 快速原型 | 全局一致 |

**推荐策略**：
1. 快速探索 → 使用 Ontogenia
2. 生产环境 → 根据规模选择 Memoryless 或 CQbyCQ
3. 最佳质量 → 运行所有三种方法并合并结果

---

## 相关文档

- [本体生成完整指南](ONTO_GENERATION_GUIDE.md)
- [示例代码](examples/cq_ontology_methods_examples.py)
- [图构建指南](GRAPH_CONSTRUCT_GUIDE.md)
