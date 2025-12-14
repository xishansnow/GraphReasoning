# Triple Extractor 快速参考指南

## 概述

`triple_extractor` 模块提供了一个简化的接口，用于直接从文本块中提取知识图谱三元组（主语-谓语-宾语），而无需完整的图构建流程。

## 核心特性

✅ **直接提取**：从文本块直接提取三元组  
✅ **批量处理**：支持多个文本块的批量提取  
✅ **Schema 验证**：可选的本体/模式验证  
✅ **实体规范化**：自动规范化实体名称  
✅ **进度跟踪**：内置进度条显示  
✅ **灵活输出**：支持列表、DataFrame、JSON 格式  

---

## 快速开始

### 1. 基本导入

```python
from GraphConstruct import (
    TripleExtractor,
    extract_triples_from_chunk,
    extract_triples_from_chunks,
    extract_triples_to_dataframe,
    save_triples,
    load_triples,
)

from Llms.llm_providers import get_generator
```

### 2. 初始化 LLM 生成器

```python
# OpenAI
generate = get_generator("openai", model="gpt-4")

# 或者本地模型
generate = get_generator("local", model="mistral-openorca:latest")
```

---

## 使用方法

### 方法 1：便捷函数（推荐用于简单场景）

#### 单个文本块提取

```python
chunk = "Albert Einstein developed the theory of relativity in 1915."

triples = extract_triples_from_chunk(
    chunk=chunk,
    generate=generate,
    normalize=True,      # 规范化实体名称
    validate=False,      # 不验证（更快）
    verbatim=False       # 不显示详细日志
)

# 输出示例
# [
#   {
#     'node_1': 'albert einstein',
#     'node_2': 'theory of relativity', 
#     'edge': 'developed',
#     'chunk_id': '...'
#   }
# ]
```

#### 多个文本块提取

```python
chunks = [
    "Einstein developed relativity.",
    "Newton discovered gravity.",
    "Darwin proposed evolution."
]

triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    show_progress=True,  # 显示进度条
    normalize=True
)

print(f"提取了 {len(triples)} 个三元组")
```

#### 提取到 DataFrame

```python
df = extract_triples_to_dataframe(
    chunks=chunks,
    generate=generate,
    normalize=True,
    show_progress=True
)

# DataFrame 包含列: node_1, node_2, edge, chunk_id
print(df.head())
```

### 方法 2：使用 TripleExtractor 类（推荐用于复杂场景）

```python
# 创建提取器实例
extractor = TripleExtractor(
    generate=generate,
    schema=None,              # 可选：Schema 对象
    normalize_entities=True,  # 规范化实体
    validate_triples=False,   # 验证三元组
    verbatim=False            # 详细输出
)

# 提取单个块
triples = extractor.extract_from_chunk(
    chunk="Your text here...",
    chunk_id="unique_id",     # 可选
    repeat_refine=0,          # 精化迭代次数
    metadata={"source": "book1"}  # 附加元数据
)

# 提取多个块
triples = extractor.extract_from_chunks(
    chunks=chunk_list,
    chunk_ids=id_list,        # 可选
    repeat_refine=0,
    show_progress=True,
    metadata=metadata_list    # 每个块的元数据
)

# 提取到 DataFrame
df = extractor.extract_to_dataframe(
    chunks=chunk_list,
    repeat_refine=0,
    show_progress=True
)
```

---

## 高级功能

### 1. Schema 验证

```python
from GraphConstruct import GraphSchema

# 定义 Schema
schema = GraphSchema(
    entity_types={
        "Person": {"properties": ["name", "age"]},
        "Organization": {"properties": ["name", "sector"]},
        "Location": {"properties": ["name", "country"]},
    },
    relation_types={
        "works_for": {"domain": "Person", "range": "Organization"},
        "located_in": {"domain": ["Person", "Organization"], "range": "Location"},
    }
)

# 使用 Schema 验证提取
extractor = TripleExtractor(
    generate=generate,
    schema=schema,
    validate_triples=True,  # 启用验证
    verbatim=True           # 显示被拒绝的三元组
)

triples = extractor.extract_from_chunk(chunk)
# 只返回符合 Schema 的三元组
```

### 2. 迭代精化提取

```python
# repeat_refine=0: 快速提取，质量一般
# repeat_refine=1: 平衡速度与质量
# repeat_refine=2+: 高质量，但较慢

triples = extract_triples_from_chunk(
    chunk=long_complex_text,
    generate=generate,
    repeat_refine=2,  # 两次精化迭代
    verbatim=True     # 显示每次迭代结果
)
```

### 3. 附加元数据

```python
# 相同元数据应用于所有块
metadata = {"source": "textbook", "chapter": 5}

triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    metadata=metadata  # 字典：应用到所有块
)

# 每个块不同的元数据
metadata_list = [
    {"source": "book1", "page": 10},
    {"source": "book1", "page": 11},
    {"source": "book2", "page": 5},
]

triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    metadata=metadata_list  # 列表：每个块一个
)

# 访问元数据
for triple in triples:
    print(f"来源: {triple['source']}, 页码: {triple['page']}")
```

### 4. 保存和加载三元组

```python
# 保存为 CSV
save_triples(triples, "output/triples.csv", format="csv")

# 保存为 JSON
save_triples(triples, "output/triples.json", format="json")

# 保存为 JSONL（每行一个 JSON 对象）
save_triples(triples, "output/triples.jsonl", format="jsonl")

# 加载三元组
df = load_triples("output/triples.csv")  # 自动检测格式
df = load_triples("output/triples.json", format="json")  # 显式指定
```

---

## 参数说明

### TripleExtractor 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `generate` | callable | *必需* | LLM 生成函数 |
| `schema` | GraphSchema | None | Schema 对象用于验证 |
| `normalize_entities` | bool | True | 是否规范化实体名称 |
| `validate_triples` | bool | False | 是否验证三元组 |
| `verbatim` | bool | False | 是否显示详细日志 |

### extract_from_chunk 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk` | str | *必需* | 要提取的文本块 |
| `chunk_id` | str | None | 块标识符（自动生成） |
| `repeat_refine` | int | 0 | 精化迭代次数 |
| `metadata` | dict | None | 附加元数据 |

### extract_from_chunks 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunks` | List[str] | *必需* | 文本块列表 |
| `chunk_ids` | List[str] | None | 块标识符列表 |
| `repeat_refine` | int | 0 | 精化迭代次数 |
| `show_progress` | bool | True | 显示进度条 |
| `metadata` | dict/List[dict] | None | 元数据 |

---

## 输出格式

### 三元组字典结构

```python
{
    'node_1': 'albert einstein',      # 主语实体
    'node_2': 'theory of relativity',  # 宾语实体
    'edge': 'developed',               # 谓语/关系
    'chunk_id': 'abc123...',           # 来源块标识
    # 可选：用户定义的元数据
    'source': 'textbook',
    'page': 42,
    # 可选：实体类型（如果启用 Schema）
    'node_1_type': 'Person',
    'node_2_type': 'Concept'
}
```

### DataFrame 列

| 列名 | 类型 | 说明 |
|------|------|------|
| `node_1` | str | 主语实体 |
| `node_2` | str | 宾语实体 |
| `edge` | str | 谓语/关系 |
| `chunk_id` | str | 来源块标识 |
| `node_1_type` | str | 主语类型（可选） |
| `node_2_type` | str | 宾语类型（可选） |
| 其他 | 任意 | 用户定义的元数据 |

---

## 最佳实践

### ✅ 推荐做法

1. **启用规范化**：总是使用 `normalize_entities=True` 以去除重复
2. **批量处理**：对多个块使用 `extract_from_chunks` 而不是循环调用
3. **显示进度**：长时间任务启用 `show_progress=True`
4. **保存中间结果**：定期保存提取的三元组避免数据丢失
5. **使用 Schema**：对于结构化知识图谱，定义并使用 Schema
6. **适度精化**：`repeat_refine=1` 通常足够，更高值回报递减

### ❌ 避免做法

1. 不要对每个块单独调用而不使用批量方法
2. 不要在不需要时设置 `verbatim=True`（会产生大量输出）
3. 不要在简单提取中使用过高的 `repeat_refine` 值
4. 不要忘记处理空块或错误情况

---

## 性能优化

### 速度优化

```python
# 最快速度（降低质量）
triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    repeat_refine=0,        # 无精化
    normalize=False,        # 无规范化
    validate=False,         # 无验证
    show_progress=True
)
```

### 质量优化

```python
# 最高质量（降低速度）
from GraphConstruct import GraphSchema

schema = GraphSchema(...)  # 定义详细 Schema

extractor = TripleExtractor(
    generate=generate,
    schema=schema,
    normalize_entities=True,   # 规范化
    validate_triples=True,     # 验证
    verbatim=True              # 查看被拒绝的三元组
)

triples = extractor.extract_from_chunks(
    chunks=chunks,
    repeat_refine=2,           # 两次精化
    show_progress=True
)
```

### 平衡方案（推荐）

```python
# 速度与质量平衡
triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    repeat_refine=1,        # 一次精化
    normalize=True,         # 规范化
    validate=False,         # 无验证（或按需）
    show_progress=True
)
```

---

## 与 graph_generation 对比

### graph_generation.py（完整流程）

```python
from GraphConstruct import make_graph_from_text

# 完整的知识图谱构建流程
graph_html, graph_graphml, G, net, pdf, stats = make_graph_from_text(
    txt=long_text,
    generate=generate,
    chunk_size=2500,
    include_contextual_proximity=True,
    # ... 更多参数
)
```

**特点**：
- ✅ 自动文本分块
- ✅ 三元组提取
- ✅ 图构建
- ✅ 社区检测
- ✅ 可视化
- ❌ 较重量级，不灵活

### triple_extractor.py（仅提取）

```python
from GraphConstruct import extract_triples_from_chunks

# 仅提取三元组
triples = extract_triples_from_chunks(
    chunks=chunks,  # 你自己的分块
    generate=generate,
    show_progress=True
)

# 你可以自定义后续处理
```

**特点**：
- ✅ 轻量级
- ✅ 灵活控制
- ✅ 可集成到自定义流程
- ✅ 直接访问三元组数据
- ❌ 需要手动分块
- ❌ 不包含可视化

---

## 常见问题

### Q1: 何时使用 triple_extractor vs graph_generation?

**使用 triple_extractor 当你需要：**
- 仅提取三元组，不构建完整图
- 集成到自定义工作流
- 批量处理大量文档
- 对提取过程精细控制

**使用 graph_generation 当你需要：**
- 端到端知识图谱构建
- 自动可视化
- 社区检测和分析
- 快速原型开发

### Q2: 如何处理大型文档？

```python
# 方法 1：分批处理
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = splitter.split_text(large_document)

# 分批提取（每批 50 个块）
batch_size = 50
all_triples = []

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    triples = extract_triples_from_chunks(batch, generate)
    all_triples.extend(triples)
    
    # 定期保存
    if (i // batch_size) % 10 == 0:
        save_triples(all_triples, f"backup_{i}.csv")
```

### Q3: 如何提高提取质量？

1. **使用更好的模型**：GPT-4 > GPT-3.5 > 本地模型
2. **增加精化迭代**：`repeat_refine=1` 或 `2`
3. **使用 Schema 验证**：过滤无效三元组
4. **清理输入文本**：移除噪声、格式化
5. **提供示例**：在提示模板中添加示例（修改 prompt_templates.py）

### Q4: 如何自定义提取格式？

修改 `Llms/prompt_templates.py` 中的提示模板：

```python
# 找到 PROMPT_GRAPH_MAKER_INITIAL 并修改
# 添加你自己的指令和示例
```

---

## 错误处理

```python
from GraphConstruct import TripleExtractor

extractor = TripleExtractor(generate=generate, verbatim=False)

all_triples = []
errors = []

for i, chunk in enumerate(chunks):
    try:
        triples = extractor.extract_from_chunk(
            chunk=chunk,
            chunk_id=f"chunk_{i}"
        )
        
        if triples:
            all_triples.extend(triples)
            print(f"✅ 块 {i}: {len(triples)} 个三元组")
        else:
            print(f"⚠️  块 {i}: 未提取到三元组")
            errors.append((i, "No triples extracted"))
            
    except Exception as e:
        print(f"❌ 块 {i}: 错误 - {e}")
        errors.append((i, str(e)))

print(f"\n总计: {len(all_triples)} 个三元组, {len(errors)} 个错误")
```

---

## 示例代码

完整示例请参考：
- `examples/triple_extractor_examples.py` - 8 个详细示例
- `examples/graph_construct_examples.py` - 图构建示例

---

## 总结

`triple_extractor` 模块提供了：

✅ **简单快速**的三元组提取  
✅ **灵活可控**的处理流程  
✅ **批量处理**能力  
✅ **Schema 验证**支持  
✅ **多种输出**格式  

非常适合需要从文本块直接提取知识图谱三元组的场景！
