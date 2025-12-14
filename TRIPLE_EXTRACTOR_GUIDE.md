# Triple Extractor 模块使用指南

## 目录

1. [概述](#概述)
2. [设计理念](#设计理念)
3. [核心功能](#核心功能)
4. [架构设计](#架构设计)
5. [使用方法](#使用方法)
6. [高级特性](#高级特性)
7. [性能优化](#性能优化)
8. [完整示例](#完整示例)
9. [常见问题](#常见问题)
10. [API 参考](#api-参考)

---

## 概述

`triple_extractor` 模块是 GraphConstruct 包的一部分，专门用于从文本块直接提取知识图谱三元组（subject-predicate-object）。与完整的 `graph_generation` 流程相比，该模块提供了更轻量级、更灵活的三元组提取接口。

### 关键特性

- ✅ **直接提取**：无需完整图构建流程，直接从文本提取三元组
- ✅ **批量处理**：支持高效的批量文本块处理
- ✅ **Schema 验证**：可选的本体/模式验证确保数据质量
- ✅ **实体规范化**：自动消除实体名称不一致问题
- ✅ **灵活输出**：支持列表、DataFrame、JSON 等多种格式
- ✅ **进度跟踪**：内置进度条，实时监控处理进度
- ✅ **元数据支持**：可为每个三元组附加自定义元数据

---

## 设计理念

### 为什么需要 triple_extractor？

在知识图谱构建中，有时我们只需要提取三元组，而不需要完整的图构建、可视化和分析流程。`triple_extractor` 模块专为这种场景设计：

**传统方法（graph_generation.py）：**
```
文本 → 自动分块 → 三元组提取 → 图构建 → 社区检测 → 可视化
```

**triple_extractor 方法：**
```
文本块 → 三元组提取 → 完成
```

### 设计原则

1. **单一职责**：专注于三元组提取，不包含图构建逻辑
2. **灵活性**：用户完全控制输入和输出
3. **可组合性**：易于集成到自定义工作流
4. **高效性**：批量处理优化，支持大规模数据
5. **兼容性**：与现有 graph_generation 模块共享核心提取逻辑

---

## 核心功能

### 1. TripleExtractor 类

核心提取器类，提供完整的三元组提取功能：

```python
from GraphConstruct import TripleExtractor

extractor = TripleExtractor(
    generate=llm_generate_fn,      # LLM 生成函数
    schema=None,                    # 可选 Schema
    normalize_entities=True,        # 实体规范化
    validate_triples=False,         # Schema 验证
    verbatim=False                  # 详细日志
)
```

### 2. 便捷函数

为常见场景提供的快捷函数：

```python
# 单个文本块
extract_triples_from_chunk(chunk, generate, ...)

# 多个文本块
extract_triples_from_chunks(chunks, generate, ...)

# 直接输出 DataFrame
extract_triples_to_dataframe(chunks, generate, ...)
```

### 3. I/O 工具

保存和加载三元组：

```python
# 保存
save_triples(triples, filepath, format="csv")

# 加载
triples = load_triples(filepath)
```

---

## 架构设计

### 模块结构

```
GraphConstruct/
├── triple_extractor.py          # 主模块
│   ├── TripleExtractor          # 核心类
│   ├── extract_triples_from_chunk      # 便捷函数
│   ├── extract_triples_from_chunks     # 批量函数
│   ├── extract_triples_to_dataframe    # DataFrame 输出
│   ├── save_triples             # 保存功能
│   └── load_triples             # 加载功能
│
├── graph_generation.py          # 共享功能
│   ├── GraphSchema              # Schema 定义
│   ├── validate_and_filter_triples    # 验证函数
│   └── normalize_entity_names   # 规范化函数
│
└── __init__.py                  # 包导出
```

### 提取流程

```
输入文本块
    ↓
[步骤 1] LLM 初始提取
    ↓
[步骤 2] 格式改进
    ↓
[步骤 3] 格式修复
    ↓
[步骤 4] 可选：迭代精化 (repeat_refine)
    ↓
[步骤 5] JSON 解析
    ↓
[可选] 实体规范化
    ↓
[可选] Schema 验证
    ↓
输出三元组列表
```

### 与其他模块的关系

```
triple_extractor
    ↓ 调用
Llms.prompt_templates (获取提示模板)
    ↓ 使用
graph_generation.GraphSchema (Schema 定义)
graph_generation.validate_and_filter_triples (验证)
graph_generation.normalize_entity_names (规范化)
```

---

## 使用方法

### 基础用法

#### 1. 初始化 LLM 生成器

```python
from Llms.llm_providers import get_generate_fn

# OpenAI
generate = get_generate_fn("openai", config={"model": "gpt-4"})

# 本地模型
generate = get_generate_fn("local", config={"model": "mistral:latest"})
```

#### 2. 提取单个文本块

```python
from GraphConstruct import extract_triples_from_chunk

chunk = """
Albert Einstein was born in Ulm, Germany in 1879. 
He developed the theory of relativity and won the 
Nobel Prize in Physics in 1921.
"""

triples = extract_triples_from_chunk(
    chunk=chunk,
    generate=generate,
    normalize=True,
    verbatim=True
)

# 输出
for triple in triples:
    print(f"{triple['node_1']} --[{triple['edge']}]--> {triple['node_2']}")
```

**输出示例：**
```
albert einstein --[born_in]--> ulm
albert einstein --[developed]--> theory of relativity
albert einstein --[won]--> nobel prize in physics
```

#### 3. 批量提取多个文本块

```python
from GraphConstruct import extract_triples_from_chunks

chunks = [
    "Newton discovered the law of gravity.",
    "Darwin proposed the theory of evolution.",
    "Marie Curie won two Nobel Prizes."
]

triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    show_progress=True
)

print(f"提取了 {len(triples)} 个三元组")
```

#### 4. 输出为 DataFrame

```python
from GraphConstruct import extract_triples_to_dataframe

df = extract_triples_to_dataframe(
    chunks=chunks,
    generate=generate,
    normalize=True,
    show_progress=True
)

print(df.head())
```

**DataFrame 输出：**
```
     node_1              edge                node_2        chunk_id
0    newton      discovered           law of gravity    abc123...
1    darwin        proposed     theory of evolution    def456...
2  marie curie       won         two nobel prizes      ghi789...
```

---

## 高级特性

### 1. Schema 验证

定义知识图谱的结构约束，确保提取的三元组符合预期模式。

```python
from GraphConstruct import GraphSchema, TripleExtractor

# 定义 Schema
schema = GraphSchema(
    entity_types={
        "Person": {
            "properties": ["name", "birthdate", "nationality"]
        },
        "Organization": {
            "properties": ["name", "founded_date", "sector"]
        },
        "Location": {
            "properties": ["name", "country", "type"]
        },
        "Concept": {
            "properties": ["name", "definition"]
        }
    },
    relation_types={
        "born_in": {
            "domain": "Person",
            "range": "Location"
        },
        "works_for": {
            "domain": "Person",
            "range": "Organization"
        },
        "located_in": {
            "domain": ["Person", "Organization"],
            "range": "Location"
        },
        "developed": {
            "domain": "Person",
            "range": "Concept"
        }
    }
)

# 使用 Schema 验证
extractor = TripleExtractor(
    generate=generate,
    schema=schema,
    normalize_entities=True,
    validate_triples=True,  # 启用验证
    verbatim=True           # 显示被拒绝的三元组
)

triples = extractor.extract_from_chunk(chunk)
```

**验证输出示例：**
```
❌ Rejected: (einstein) --[invented]--> (time machine)
   Reason: Invalid predicate: invented

✅ Valid triples: 5
❌ Invalid triples: 2
```

### 2. 迭代精化提取

通过多次迭代提高提取质量：

```python
triples = extract_triples_from_chunk(
    chunk=complex_text,
    generate=generate,
    repeat_refine=2,  # 进行 2 次精化
    verbatim=True
)
```

**精化过程：**
1. **初始提取**：快速提取主要三元组
2. **第 1 次精化**：
   - 添加遗漏的三元组
   - 优化已有三元组
   - 修复格式问题
3. **第 2 次精化**：
   - 进一步完善
   - 消除冗余
   - 最终质量检查

**性能对比：**
| repeat_refine | 提取时间 | 三元组数量 | 质量评分 |
|---------------|----------|------------|----------|
| 0             | 1x       | 基准       | ★★★☆☆    |
| 1             | 2x       | +20%       | ★★★★☆    |
| 2             | 3x       | +30%       | ★★★★★    |

### 3. 元数据附加

为每个三元组附加上下文信息：

```python
# 方式 1: 所有块使用相同元数据
metadata = {
    "source": "history_textbook",
    "chapter": 5,
    "author": "John Doe"
}

triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    metadata=metadata
)

# 方式 2: 每个块使用不同元数据
metadata_list = [
    {"source": "book1", "page": 10},
    {"source": "book1", "page": 11},
    {"source": "book2", "page": 5}
]

triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    metadata=metadata_list
)

# 访问元数据
for triple in triples:
    print(f"来源: {triple['source']}, 页码: {triple['page']}")
    print(f"三元组: {triple['node_1']} --[{triple['edge']}]--> {triple['node_2']}")
```

### 4. 自定义 chunk_id

为每个文本块指定唯一标识符：

```python
# 自动生成 ID
triples = extract_triples_from_chunks(chunks=chunks, generate=generate)
# chunk_id 示例: 'a1b2c3d4...'

# 自定义 ID
custom_ids = ["paragraph_1", "paragraph_2", "paragraph_3"]

triples = extract_triples_from_chunks(
    chunks=chunks,
    chunk_ids=custom_ids,
    generate=generate
)
# chunk_id 示例: 'paragraph_1'
```

### 5. 保存和加载

```python
# 保存为 CSV（推荐用于分析）
save_triples(triples, "output/triples.csv", format="csv")

# 保存为 JSON（推荐用于交换）
save_triples(triples, "output/triples.json", format="json")

# 保存为 JSONL（推荐用于流式处理）
save_triples(triples, "output/triples.jsonl", format="jsonl")

# 加载（自动检测格式）
df = load_triples("output/triples.csv")
```

---

## 性能优化

### 速度优化策略

#### 1. 禁用可选功能

```python
# 最快速度配置
triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate,
    repeat_refine=0,        # 不精化
    normalize=False,        # 不规范化
    validate=False,         # 不验证
    show_progress=True
)
```

**速度提升：~3x**

#### 2. 批量处理优化

```python
# 不推荐：逐个处理
all_triples = []
for chunk in chunks:
    triples = extract_triples_from_chunk(chunk, generate)
    all_triples.extend(triples)

# 推荐：批量处理
all_triples = extract_triples_from_chunks(chunks, generate)
```

**速度提升：~1.5x**（减少函数调用开销）

#### 3. 使用更快的模型

```python
# 慢但质量高
generate_gpt4 = get_generate_fn("openai", config={"model": "gpt-4"})

# 快但质量略低
generate_gpt35 = get_generate_fn("openai", config={"model": "gpt-3.5-turbo"})

# 本地模型（最快，但需要硬件）
generate_local = get_generate_fn("local", config={"model": "mistral:latest"})
```

### 质量优化策略

#### 1. 启用所有质量功能

```python
from GraphConstruct import GraphSchema, TripleExtractor

# 定义详细 Schema
schema = GraphSchema(...)

# 质量优先配置
extractor = TripleExtractor(
    generate=generate_gpt4,     # 使用最好的模型
    schema=schema,
    normalize_entities=True,    # 规范化
    validate_triples=True,      # 验证
    verbatim=True               # 查看详细输出
)

triples = extractor.extract_from_chunks(
    chunks=chunks,
    repeat_refine=2,            # 两次精化
    show_progress=True
)
```

#### 2. 文本预处理

```python
def preprocess_chunk(text: str) -> str:
    """预处理文本以提高提取质量。"""
    # 移除多余空白
    text = " ".join(text.split())
    
    # 修复常见拼写错误
    text = text.replace("teh", "the")
    
    # 标准化标点
    text = text.replace("...", ".")
    
    return text

# 应用预处理
cleaned_chunks = [preprocess_chunk(c) for c in chunks]
triples = extract_triples_from_chunks(cleaned_chunks, generate)
```

#### 3. 分块策略优化

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 不推荐：块太小，上下文不足
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

# 推荐：适中块大小，带重叠
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # 足够的上下文
    chunk_overlap=200   # 保持连续性
)

chunks = splitter.split_text(document)
triples = extract_triples_from_chunks(chunks, generate)
```

### 平衡策略（推荐）

```python
# 速度与质量平衡的最佳实践
triples = extract_triples_from_chunks(
    chunks=chunks,
    generate=generate_gpt35,    # GPT-3.5 平衡速度和质量
    repeat_refine=1,            # 一次精化
    normalize=True,             # 规范化（重要）
    validate=False,             # 验证可选
    show_progress=True
)
```

---

## 完整示例

### 示例 1：构建历史事件知识图谱

```python
from GraphConstruct import (
    TripleExtractor,
    GraphSchema,
    save_triples
)
from Llms.llm_providers import get_generate_fn
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 定义历史事件 Schema
schema = GraphSchema(
    entity_types={
        "Person": {"properties": ["name", "role"]},
        "Event": {"properties": ["name", "date"]},
        "Location": {"properties": ["name", "country"]},
        "Organization": {"properties": ["name", "type"]}
    },
    relation_types={
        "participated_in": {"domain": "Person", "range": "Event"},
        "led": {"domain": "Person", "range": "Event"},
        "occurred_in": {"domain": "Event", "range": "Location"},
        "founded": {"domain": "Person", "range": "Organization"}
    }
)

# 2. 准备文本
historical_text = """
The French Revolution began in 1789 with the storming of the Bastille.
Louis XVI was King of France during this period. The Revolution led to
the establishment of the First French Republic in 1792. Napoleon Bonaparte
rose to power and became Emperor in 1804.
"""

# 3. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_text(historical_text)

# 4. 初始化提取器
generate = get_generate_fn("openai", config={"model": "gpt-4"})

extractor = TripleExtractor(
    generate=generate,
    schema=schema,
    normalize_entities=True,
    validate_triples=True,
    verbatim=True
)

# 5. 提取三元组
triples = extractor.extract_from_chunks(
    chunks=chunks,
    repeat_refine=1,
    show_progress=True,
    metadata={"source": "history_book", "topic": "french_revolution"}
)

# 6. 保存结果
save_triples(triples, "output/french_revolution_kg.csv")

print(f"✅ 提取了 {len(triples)} 个三元组")
```

### 示例 2：批量处理大型文档集

```python
import os
from pathlib import Path
from tqdm import tqdm

def process_document_batch(doc_paths, output_dir, batch_size=50):
    """批量处理多个文档，定期保存结果。"""
    
    generate = get_generate_fn("openai", config={"model": "gpt-3.5-turbo"})
    
    extractor = TripleExtractor(
        generate=generate,
        normalize_entities=True,
        validate_triples=False,
        verbatim=False
    )
    
    all_triples = []
    
    for doc_path in tqdm(doc_paths, desc="Processing documents"):
        # 读取文档
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        
        # 分批处理块
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # 提取三元组
            triples = extractor.extract_from_chunks(
                chunks=batch,
                show_progress=False,
                metadata={
                    "document": os.path.basename(doc_path),
                    "batch": i // batch_size
                }
            )
            
            all_triples.extend(triples)
            
            # 定期保存
            if len(all_triples) >= 1000:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(output_dir) / f"triples_backup_{timestamp}.csv"
                save_triples(all_triples, save_path)
                all_triples = []
    
    # 保存剩余三元组
    if all_triples:
        final_path = Path(output_dir) / "triples_final.csv"
        save_triples(all_triples, final_path)
    
    print("✅ 批量处理完成")

# 使用
doc_paths = [
    "data/document1.txt",
    "data/document2.txt",
    "data/document3.txt"
]

process_document_batch(doc_paths, "output/triples")
```

### 示例 3：增量知识图谱构建

```python
from GraphConstruct import load_triples, save_triples
import pandas as pd

def incremental_kg_update(new_chunks, existing_triples_path, output_path):
    """增量更新知识图谱三元组。"""
    
    # 1. 加载现有三元组
    try:
        existing_df = load_triples(existing_triples_path)
        print(f"📂 加载了 {len(existing_df)} 个现有三元组")
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        print("📂 未找到现有三元组，将创建新文件")
    
    # 2. 提取新三元组
    generate = get_generate_fn("openai", config={"model": "gpt-3.5-turbo"})
    
    new_triples = extract_triples_from_chunks(
        chunks=new_chunks,
        generate=generate,
        normalize=True,
        show_progress=True
    )
    
    new_df = pd.DataFrame(new_triples)
    print(f"✅ 提取了 {len(new_df)} 个新三元组")
    
    # 3. 合并去重
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # 基于 (node_1, edge, node_2) 去重
        combined_df = combined_df.drop_duplicates(
            subset=['node_1', 'edge', 'node_2'],
            keep='first'
        )
        
        print(f"📊 合并后共 {len(combined_df)} 个唯一三元组")
    else:
        combined_df = new_df
    
    # 4. 保存更新后的三元组
    save_triples(combined_df, output_path)
    print(f"💾 保存到 {output_path}")
    
    return combined_df

# 使用
new_chunks = [
    "New information about Einstein...",
    "Recent discoveries about quantum physics..."
]

updated_kg = incremental_kg_update(
    new_chunks=new_chunks,
    existing_triples_path="output/kg_triples.csv",
    output_path="output/kg_triples_updated.csv"
)
```

---

## 常见问题

### Q1: triple_extractor 和 graph_generation 有什么区别？

**triple_extractor：**
- ✅ 只提取三元组
- ✅ 轻量级，灵活
- ✅ 适合自定义工作流
- ❌ 需要手动分块
- ❌ 不包含可视化

**graph_generation：**
- ✅ 端到端流程
- ✅ 自动分块
- ✅ 包含可视化
- ❌ 较重量级
- ❌ 灵活性较低

### Q2: 如何提高提取质量？

1. **使用更好的模型**：GPT-4 > GPT-3.5
2. **增加精化次数**：`repeat_refine=1` 或 `2`
3. **启用 Schema 验证**：过滤无效三元组
4. **预处理文本**：清理噪声，标准化格式
5. **优化分块大小**：2000-3000 字符，带 200 字符重叠

### Q3: 如何处理大型文档？

```python
# 分批处理策略
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

### Q4: 提取速度太慢怎么办？

1. **使用更快的模型**：GPT-3.5 或本地模型
2. **禁用精化**：`repeat_refine=0`
3. **禁用验证**：`validate=False`
4. **增大分块大小**：减少 API 调用次数
5. **并行处理**：使用多线程/多进程（需要自己实现）

### Q5: 如何自定义提取格式？

修改 `Llms/prompt_templates.py` 中的提示模板：

```python
PROMPT_GRAPH_MAKER_INITIAL = PromptTemplate(
    name="graph_maker_initial",
    system_prompt="""你的自定义系统提示...""",
    user_prompt="""你的自定义用户提示...
    
    示例输出：
    [
        {"node_1": "...", "node_2": "...", "edge": "..."},
        ...
    ]
    """
)
```

### Q6: 如何处理提取错误？

```python
extractor = TripleExtractor(generate=generate, verbatim=False)

all_triples = []
errors = []

for i, chunk in enumerate(chunks):
    try:
        triples = extractor.extract_from_chunk(chunk, chunk_id=f"chunk_{i}")
        if triples:
            all_triples.extend(triples)
        else:
            errors.append((i, "No triples extracted"))
    except Exception as e:
        errors.append((i, str(e)))

# 分析错误
if errors:
    print(f"⚠️  {len(errors)} 个块提取失败")
    for idx, err in errors[:5]:  # 显示前 5 个错误
        print(f"  块 {idx}: {err}")
```

---

## API 参考

### TripleExtractor 类

```python
class TripleExtractor:
    def __init__(
        self,
        generate: Callable,
        schema: GraphSchema = None,
        normalize_entities: bool = True,
        validate_triples: bool = False,
        verbatim: bool = False
    )
```

**参数：**
- `generate`：LLM 生成函数
- `schema`：可选的 Schema 对象
- `normalize_entities`：是否规范化实体名称
- `validate_triples`：是否验证三元组
- `verbatim`：是否输出详细日志

**方法：**

#### extract_from_chunk

```python
def extract_from_chunk(
    self,
    chunk: str,
    chunk_id: str = None,
    repeat_refine: int = 0,
    metadata: dict = None
) -> List[Dict]
```

从单个文本块提取三元组。

**参数：**
- `chunk`：文本块
- `chunk_id`：块标识符
- `repeat_refine`：精化迭代次数
- `metadata`：附加元数据

**返回：**三元组字典列表

#### extract_from_chunks

```python
def extract_from_chunks(
    self,
    chunks: List[str],
    chunk_ids: List[str] = None,
    repeat_refine: int = 0,
    show_progress: bool = True,
    metadata: Union[dict, List[dict]] = None
) -> List[Dict]
```

从多个文本块提取三元组。

**参数：**
- `chunks`：文本块列表
- `chunk_ids`：块标识符列表
- `repeat_refine`：精化迭代次数
- `show_progress`：显示进度条
- `metadata`：元数据（字典或列表）

**返回：**三元组字典列表

#### extract_to_dataframe

```python
def extract_to_dataframe(
    self,
    chunks: Union[str, List[str]],
    **kwargs
) -> pd.DataFrame
```

提取三元组并返回 DataFrame。

**参数：**
- `chunks`：单个块或块列表
- `**kwargs`：传递给 extract 方法的参数

**返回：**DataFrame

### 便捷函数

#### extract_triples_from_chunk

```python
def extract_triples_from_chunk(
    chunk: str,
    generate: Callable,
    chunk_id: str = None,
    schema: GraphSchema = None,
    validate: bool = False,
    normalize: bool = True,
    repeat_refine: int = 0,
    verbatim: bool = False
) -> List[Dict]
```

#### extract_triples_from_chunks

```python
def extract_triples_from_chunks(
    chunks: List[str],
    generate: Callable,
    chunk_ids: List[str] = None,
    schema: GraphSchema = None,
    validate: bool = False,
    normalize: bool = True,
    repeat_refine: int = 0,
    show_progress: bool = True,
    verbatim: bool = False
) -> List[Dict]
```

#### extract_triples_to_dataframe

```python
def extract_triples_to_dataframe(
    chunks: Union[str, List[str]],
    generate: Callable,
    schema: GraphSchema = None,
    validate: bool = False,
    normalize: bool = True,
    repeat_refine: int = 0,
    show_progress: bool = True,
    verbatim: bool = False
) -> pd.DataFrame
```

### I/O 函数

#### save_triples

```python
def save_triples(
    triples: Union[List[Dict], pd.DataFrame],
    filepath: Union[str, Path],
    format: str = "csv"
) -> None
```

**格式：**`"csv"`, `"json"`, `"jsonl"`

#### load_triples

```python
def load_triples(
    filepath: Union[str, Path],
    format: str = None
) -> pd.DataFrame
```

**格式：**自动检测或显式指定

---

## 总结

`triple_extractor` 模块提供了：

✅ **简单快速**的三元组提取接口  
✅ **灵活可控**的处理流程  
✅ **批量处理**能力  
✅ **Schema 验证**支持  
✅ **多种输出**格式  
✅ **完善的文档**和示例  

非常适合需要从文本块直接提取知识图谱三元组的各种场景！

---

**相关文档：**
- [快速参考指南](TRIPLE_EXTRACTOR_QUICK_REFERENCE.md)
- [完整示例](examples/triple_extractor_examples.py)
- [Graph Generation 指南](GRAPH_CONSTRUCT_GUIDE.md)
