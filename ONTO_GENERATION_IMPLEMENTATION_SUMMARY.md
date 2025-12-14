# onto_generation 模块实现总结

## 📦 模块信息

**位置**: `GraphConstruct/onto_generation.py`

**版本**: v1.0

**依赖**: 仅标准库 (无外部依赖)

---

## 🎯 功能概述

`onto_generation` 模块提供**两种方法**自动生成和归纳本体/Schema：

### 方法 1: 自上而下（Top-Down）
- **输入**: 能力问题（Competency Questions）
- **过程**: 从需求出发，定义系统应该支持的概念
- **输出**: 完整的本体框架
- **适用**: 系统设计初期，需求明确

### 方法 2: 自下而上（Bottom-Up）
- **输入**: 现有的 Triples 数据
- **过程**: 从数据中分析和归纳规律
- **输出**: 数据驱动的本体
- **适用**: 数据探索，本体发现

---

## 🏗️ 核心组件

### 1. TripleAnalyzer 类

**功能**: 分析 Triple 数据，提取实体和关系模式

**主要方法**:
- `analyze_triples()` - 统计分析
- `infer_entity_types()` - 推导实体类型
- `infer_relation_types()` - 推导关系类型和 domain/range

**输出示例**:
```python
{
    "total_entities": 156,
    "total_relations": 42,
    "entity_distribution": {"Person": 42, "Organization": 28, ...},
    "relation_distribution": {"works_for": 25, "located_in": 12, ...}
}
```

### 2. TopDownOntologyExtractor 类

**功能**: 从能力问题提取本体

**主要方法**:
- `extract_from_competency_questions()` - 从问题列表提取本体

**工作流程**:
1. 从问题中识别实体类型（Person, Organization, Event, etc）
2. 识别关系类型和 domain/range 约束
3. 为每个实体类型推导常见属性

**输出示例**:
```python
{
    "entity_types": {
        "Person": {
            "name": "Person",
            "properties": ["name", "age", "role"]
        }
    },
    "relation_types": {
        "works_for": {
            "domain": "Person",
            "range": "Organization"
        }
    }
}
```

### 3. BottomUpOntologyInducer 类

**功能**: 从 Triples 数据归纳本体

**主要方法**:
- `induce_ontology_from_triples()` - 从 triples 列表归纳本体

**工作流程**:
1. 创建 TripleAnalyzer 实例
2. 分析 triples 数据
3. 推导实体类型和频率
4. 推导关系类型和 domain/range
5. 生成统计信息

**输出示例**:
```python
{
    "entity_types": {
        "Scholar": {
            "name": "Scholar",
            "frequency": 42,
            "examples": ["Alice", "Bob"],
            "confidence": 0.95
        }
    },
    "relation_types": {
        "works_at": {
            "name": "works_at",
            "domain": "Scholar",
            "range": "University",
            "frequency": 23,
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

### 4. OntologyMerger 类

**功能**: 合并多个本体

**策略**:
- `"union"` - 保留所有类型
- `"intersection"` - 只保留共同类型

**使用场景**:
- 合并需求驱动和数据驱动的本体
- 比对不同数据集的本体
- 协调多个信息源

---

## 🔧 便利函数

### generate_ontology_from_questions()

```python
def generate_ontology_from_questions(
    questions: List[str],
    generate_fn = None,
    verbatim: bool = False
) -> Dict
```

从能力问题快速生成本体。

### generate_ontology_from_triples()

```python
def generate_ontology_from_triples(
    triples: List[Dict],
    min_frequency: int = 2,
    verbatim: bool = False
) -> Dict
```

从 triples 快速生成本体。支持最小频率过滤。

### ontology_to_graphschema()

```python
def ontology_to_graphschema(ontology: Dict) -> GraphSchema
```

将本体转换为 GraphSchema 格式，用于三元组验证。

---

## 📊 数据结构

### Triple 数据格式

```python
{
    "node_1": str,           # 主体名称
    "node_1_type": str,      # 主体类型
    "edge": str,             # 关系名
    "node_2": str,           # 客体名称
    "node_2_type": str,      # 客体类型
}
```

### InferredEntityType 数据类

```python
@dataclass
class InferredEntityType:
    name: str                  # 实体类型名
    frequency: int            # 出现次数
    examples: List[str]       # 示例
    description: str = ""     # 描述
    properties: List[str] = None  # 属性列表
    confidence: float = 0.0   # 置信度
```

### InferredRelationType 数据类

```python
@dataclass
class InferredRelationType:
    name: str                 # 关系名
    domain: str              # 定义域
    range: str               # 值域
    frequency: int           # 出现次数
    examples: List[Tuple[str, str]] = None  # 示例
    confidence: float = 0.0  # 置信度
```

---

## 📈 测试结果

所有 5 个测试组件已验证通过 ✅：

1. **TripleAnalyzer** - 分析、推导实体/关系类型
2. **TopDownOntologyExtractor** - 从能力问题提取本体
3. **BottomUpOntologyInducer** - 从 triples 归纳本体
4. **OntologyMerger** - 合并本体（union/intersection）
5. **Convenience Functions** - 便利函数

```bash
$ python3 test_onto_generation.py
✨ ALL TESTS PASSED! ✨
```

---

## 🚀 集成到 GraphConstruct

### 包导出

所有主要类和函数都已通过 `GraphConstruct/__init__.py` 导出：

```python
from GraphConstruct import (
    # 分析工具
    TripleAnalyzer,
    TopDownOntologyExtractor,
    BottomUpOntologyInducer,
    OntologyMerger,
    # 便利函数
    generate_ontology_from_questions,
    generate_ontology_from_triples,
    ontology_to_graphschema,
    # 数据类
    InferredEntityType,
    InferredRelationType,
    EntityTypeInferenceMethod,
)
```

### 与 GraphSchema 的集成

生成的本体可直接转换为 GraphSchema，用于图生成：

```python
from GraphConstruct import (
    generate_ontology_from_questions,
    ontology_to_graphschema,
    make_graph_from_text
)

# 生成本体
onto = generate_ontology_from_questions(questions)

# 转为 schema
schema = ontology_to_graphschema(onto)

# 在图生成中使用
graph, _, _, _, _, stats = make_graph_from_text(
    text,
    generate_fn=llm,
    schema=schema,
    validate_against_schema=True
)
```

---

## 📚 文档

### 主要文档

| 文档 | 内容 |
|------|------|
| [ONTO_GENERATION_GUIDE.md](ONTO_GENERATION_GUIDE.md) | 详细使用指南，包含 8 个代码示例 |
| [ONTO_GENERATION_QUICK_REFERENCE.md](ONTO_GENERATION_QUICK_REFERENCE.md) | 快速参考，API 速查表 |
| [examples/onto_generation_examples.py](examples/onto_generation_examples.py) | 6 个完整示例程序 |

### 相关文档

- [GRAPH_SCHEMA_USAGE.md](GRAPH_SCHEMA_USAGE.md) - GraphSchema 使用指南
- [GRAPH_CONSTRUCT_GUIDE.md](GRAPH_CONSTRUCT_GUIDE.md) - GraphConstruct 包指南

---

## 💡 使用示例

### 示例 1: 快速开始

```python
from GraphConstruct import generate_ontology_from_questions

questions = [
    "Which authors wrote which books?",
    "In which genres are books classified?"
]

onto = generate_ontology_from_questions(questions, verbatim=True)
print(onto)
```

### 示例 2: 完整工作流

```python
from GraphConstruct import (
    generate_ontology_from_questions,
    generate_ontology_from_triples,
    OntologyMerger,
    ontology_to_graphschema,
    make_graph_from_text
)

# 第1步: 需求驱动的本体
onto_req = generate_ontology_from_questions(requirements_questions)

# 第2步: 数据驱动的本体
onto_data = generate_ontology_from_triples(existing_triples, min_frequency=3)

# 第3步: 合并
onto = OntologyMerger.merge_ontologies([onto_req, onto_data], strategy="union")

# 第4步: 转为 Schema
schema = ontology_to_graphschema(onto)

# 第5步: 使用
graph = make_graph_from_text(text, generate_fn=llm, schema=schema, validate_against_schema=True)
```

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `questions` | - | 能力问题列表 |
| `triples` | - | Triple 字典列表 |
| `min_frequency` | 2 | 最小出现次数阈值 |
| `verbatim` | False | 是否打印详细信息 |
| `strategy` | - | 合并策略 ("union" 或 "intersection") |
| `method` | PATTERN | 实体类型推导方法 |
| `generate_fn` | None | LLM 生成函数 |

---

## 🔍 性能特点

- ✅ **无外部依赖** - 仅使用标准库，轻量级
- ✅ **快速分析** - O(n) 时间复杂度，适合大规模数据
- ✅ **灵活参数** - min_frequency 等参数便于调整
- ✅ **独立模块** - 可单独使用，不依赖其他 GraphConstruct 组件
- ✅ **完整测试** - 所有核心功能已验证

---

## 🎓 最佳实践

1. **结合两种方法**
   - 用自上而下定义框架（需求）
   - 用自下而上验证和扩展（数据）

2. **调整 min_frequency**
   - 探索: min_frequency=1
   - 平衡: min_frequency=2
   - 严格: min_frequency=5+

3. **版本控制本体**
   - 保存为 JSON
   - 对比历史版本
   - 跟踪本体演进

4. **质量检查**
   - 审查 domain/range 约束
   - 检查置信度分数
   - 验证示例数据

---

## 📝 公开 API

### 导出的类

```python
EntityTypeInferenceMethod  # 推导方法枚举
InferredEntityType         # 实体类型数据类
InferredRelationType       # 关系类型数据类
TripleAnalyzer            # 分析工具
TopDownOntologyExtractor  # 需求驱动提取器
BottomUpOntologyInducer   # 数据驱动归纳器
OntologyMerger            # 本体合并工具
```

### 导出的函数

```python
generate_ontology_from_questions()   # 从问题生成本体
generate_ontology_from_triples()     # 从 triples 生成本体
ontology_to_graphschema()            # 转换为 GraphSchema
```

---

## 🔗 后续增强方向

1. **LLM-辅助类型推导**
   - 使用 LLM 增强实体/关系类型推导

2. **约束推导**
   - 自动推导更多 domain/range 约束
   - 基数约束 (1:1, 1:N, M:N)

3. **版本管理**
   - 本体版本控制
   - 变更跟踪

4. **可视化**
   - 本体图形化展示
   - 类型关系可视化

5. **验证增强**
   - 更多约束类型支持
   - 自动修复建议

---

## 🏁 总结

`onto_generation` 模块是一个轻量级、高效的本体生成工具，支持：

- ✅ 自上而下：从能力问题提取本体
- ✅ 自下而上：从数据归纳本体
- ✅ 本体合并：统一多个本体
- ✅ Schema 转换：直接用于图生成验证
- ✅ 无外部依赖：轻量级部署
- ✅ 完整测试：生产就绪

可广泛应用于知识图谱构建、信息抽取、数据质量控制等场景。
