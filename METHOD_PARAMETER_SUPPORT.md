# Method Parameter Support - Quick Reference

## Overview

所有主要的本体生成函数和类现在都支持 `method` 参数，允许用户选择不同的生成策略。

## 函数签名变更

### 1. `TopDownOntologyExtractor` 类

```python
# 构造器
class TopDownOntologyExtractor:
    def __init__(self, generate_fn=None, method: str = 'pattern'):
        """
        method: 'pattern' (默认), 'cqbycq', 'memoryless', 或 'ontogenia'
        """
    
    # 主方法
    def extract_from_competency_questions(self, questions, verbatim=False, method=None):
        """
        method: 可选覆盖构造器中的默认方法
        """
    
    # 新增方法
    def compare_methods(self, questions, verbatim=False):
        """比较所有四种方法"""
```

### 2. 便捷函数

#### `generate_ontology_from_questions()`
```python
def generate_ontology_from_questions(
    questions: List[str],
    generate_fn=None,
    method: str = 'pattern',  # ✨ 新参数
    verbatim: bool = False
) -> Dict:
```

#### `generate_ontology_cqbycq()`
```python
def generate_ontology_cqbycq(
    competency_questions: List[str],
    generate_fn=None,
    verbatim: bool = False,
    **kwargs  # ✨ 新增支持
) -> Dict:
```

#### `generate_ontology_memoryless()`
```python
def generate_ontology_memoryless(
    competency_questions: List[str],
    generate_fn=None,
    verbatim: bool = False,
    **kwargs  # ✨ 新增支持
) -> Dict:
```

#### `generate_ontology_ontogenia()`
```python
def generate_ontology_ontogenia(
    competency_questions: List[str],
    generate_fn=None,
    verbatim: bool = False,
    **kwargs  # ✨ 新增支持
) -> Dict:
```

#### `compare_cq_methods()` - 增强版
```python
def compare_cq_methods(
    competency_questions: List[str],
    generate_fn=None,
    methods: List[str] = None,  # ✨ 新参数，默认 ['pattern', 'cqbycq', 'memoryless', 'ontogenia']
    verbatim: bool = False
) -> Dict:
```

#### `generate_ontology_from_triples()`
```python
def generate_ontology_from_triples(
    triples: List[Dict],
    min_frequency: int = 2,
    method: str = 'frequency',  # ✨ 新参数
    verbatim: bool = False,
    **kwargs  # ✨ 新增支持
) -> Dict:
```

## 使用示例

### 示例 1: 使用 TopDownOntologyExtractor 选择方法

```python
from GraphConstruct.onto_generation import TopDownOntologyExtractor

questions = [
    "Which persons work for which organizations?",
    "Where are organizations located?"
]

# 方法 1: 构造器指定方法
extractor = TopDownOntologyExtractor(method='pattern')
onto = extractor.extract_from_competency_questions(questions)

# 方法 2: 提取时覆盖方法
extractor = TopDownOntologyExtractor(method='pattern')
onto = extractor.extract_from_competency_questions(questions, method='cqbycq')

# 方法 3: 比较所有方法
results = extractor.compare_methods(questions)
```

### 示例 2: 使用便捷函数

```python
from GraphConstruct.onto_generation import (
    generate_ontology_from_questions,
    generate_ontology_cqbycq,
    generate_ontology_memoryless,
    generate_ontology_ontogenia,
    compare_cq_methods
)

questions = [
    "Which persons work for which organizations?",
    "Where are organizations located?"
]

# 使用默认方法
onto = generate_ontology_from_questions(questions)

# 使用 CQbyCQ 方法
onto = generate_ontology_from_questions(questions, method='cqbycq')

# 比较特定方法
results = compare_cq_methods(questions, methods=['pattern', 'cqbycq'])
```

### 示例 3: 生成本体从三元组

```python
from GraphConstruct.onto_generation import generate_ontology_from_triples

triples = [
    {"node_1": "Alice", "node_2": "Bob", "edge": "knows"},
    {"node_1": "Bob", "node_2": "Company X", "edge": "works_for"}
]

# 使用频率方法
onto = generate_ontology_from_triples(triples, method='frequency')

# 使用模式方法
onto = generate_ontology_from_triples(triples, method='pattern')
```

## 支持的方法

### 顶向方法（从能力问题生成）

| 方法 | 别名 | 特点 | 最佳用途 |
|-----|------|------|--------|
| `'pattern'` | Pattern-based | 基于规则，快速，确定性 | 快速原型化，演示 |
| `'cqbycq'` | Iterative with Memory | LLM-based，迭代处理 | 小规模数据集（<20 CQ） |
| `'memoryless'` | Independent Processing | LLM-based，可并行化 | 大规模数据集（>20 CQ） |
| `'ontogenia'` | All-at-Once | LLM-based，一次性处理 | 中等规模（<15 CQ） |

### 底向方法（从三元组生成）

| 方法 | 特点 | 最佳用途 |
|------|------|--------|
| `'frequency'` | 基于出现频率 | 一般用途 |
| `'pattern'` | 基于命名模式 | 有规律的实体命名 |
| `'llm'` | LLM 推断 | 需要语义理解 |
| `'hybrid'` | 混合方法 | 综合优势 |

## 方法参数优先级

当同时指定多个方法参数时，优先级如下：

1. **提取时参数** - `extract_from_competency_questions(method=...)`
2. **构造器参数** - `TopDownOntologyExtractor(method=...)`
3. **默认值** - `'pattern'`

```python
extractor = TopDownOntologyExtractor(method='cqbycq')

# 使用 'cqbycq'
onto = extractor.extract_from_competency_questions(questions)

# 覆盖为 'memoryless'
onto = extractor.extract_from_competency_questions(questions, method='memoryless')
```

## 返回值中的元数据

所有本体现在都包含 `metadata` 字段，记录生成方法：

```python
onto = generate_ontology_from_questions(questions, method='pattern')

print(onto['metadata'])
# {
#     'method': 'pattern',
#     'description': 'Pattern-based extraction (rule-based)'
# }

onto = generate_ontology_from_triples(triples, method='frequency')

print(onto['metadata'])
# {
#     'method': 'bottom_up',
#     'inference_method': 'frequency',
#     'min_frequency': 2
# }
```

## kwargs 支持

所有 LLM-based 生成器现在支持通过 `**kwargs` 传递额外参数：

```python
onto = generate_ontology_cqbycq(
    questions,
    verbatim=True,
    max_retries=3,  # 自定义参数
    temperature=0.7  # 自定义参数
)
```

## 方法比较函数

新增 `compare_cq_methods()` 函数支持灵活的方法选择：

```python
# 比较所有方法
results = compare_cq_methods(questions)

# 比较特定方法
results = compare_cq_methods(
    questions, 
    methods=['pattern', 'cqbycq', 'ontogenia']
)

# 获取比较统计
print(results['comparison_stats'])
# {
#     'pattern_entities': 2,
#     'pattern_relations': 1,
#     'cqbycq_entities': 3,
#     'cqbycq_relations': 2,
#     ...
# }
```

## 向后兼容性

所有现有代码继续工作，无需修改：

```python
# 旧代码仍然有效
from GraphConstruct.onto_generation import generate_ontology_from_questions

questions = ["Which persons work for which organizations?"]
onto = generate_ontology_from_questions(questions)  # 使用默认 'pattern' 方法
```

## 测试

运行测试脚本验证所有 method 参数功能：

```bash
python test_method_parameter.py
```

## 总结

✅ `TopDownOntologyExtractor` 类现在支持 4 种提取方法  
✅ 所有便捷函数都添加了 `method` 参数或 `**kwargs` 支持  
✅ `compare_cq_methods()` 现在支持灵活的方法选择  
✅ `generate_ontology_from_triples()` 支持 `method` 参数  
✅ 所有返回值包含方法元数据  
✅ 完全向后兼容  
