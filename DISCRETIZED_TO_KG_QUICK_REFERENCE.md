# 离散化数据到知识图谱快速参考

## 5 分钟快速开始

### 安装

```bash
pip install networkx pandas h3
```

### 基本用法

```python
from DGGS import (
    discretize_cdl_crop_distribution,
    prepare_for_graph_reasoning
)

# 1. 离散化 CDL 数据
cdl_pixels = {...}  # 你的 CDL 像素数据
discretized_cdl = discretize_cdl_crop_distribution(cdl_pixels, level=12)

# 2. 转换为知识图谱
triplets, graph = prepare_for_graph_reasoning(
    discretized_data=discretized_cdl,
    data_type="cdl",
    output_dir="./kg"
)

print(f"✓ 已生成 {len(triplets)} 个三元组")
print(f"✓ 图谱包含 {graph.number_of_nodes()} 个节点")

# 3. 与 GraphReasoning 集成
from GraphReasoning.graph_generation import make_graph_from_text
import pandas as pd

df = pd.read_csv("./kg/cdl_triplets.csv", sep="|")
G = make_graph_from_text(df.values.tolist())

# 4. 进行知识图谱推理
# 见 GraphReasoning 文档
```

---

## 常见任务速查

### 任务 1：CDL 作物数据到知识图谱

```python
from DGGS import prepare_for_graph_reasoning

triplets, graph = prepare_for_graph_reasoning(
    discretized_data=cdl_discretized,
    data_type="cdl"
)
```

### 任务 2：SSURGO 土壤数据到知识图谱

```python
from examples.polygon_examples import discretized_ssurgo_to_triplets
from DGGS import create_knowledge_graph_from_discretized_data

# 生成三元组
triplets = []
for cell, data in ssurgo_discretized.items():
    triplets.extend(discretized_ssurgo_to_triplets(cell, data))

# 创建图
graph, _ = create_knowledge_graph_from_discretized_data(
    ssurgo_discretized, 
    data_type="ssurgo"
)
```

### 任务 3：合并 CDL 和 SSURGO 知识图谱

```python
from DGGS import merge_into_existing_graph

merged_graph = merge_into_existing_graph(
    existing_graph=cdl_graph,
    new_triplets=ssurgo_triplets,
    merge_strategy="union"
)
```

### 任务 4：导出多种格式

```python
from DGGS import (
    export_triplets_to_csv,
    export_triplets_to_json,
    export_graph_to_graphml,
    export_graph_to_rdf_turtle
)

# CSV (GraphReasoning)
export_triplets_to_csv(triplets, "triplets.csv")

# JSON (编程)
export_triplets_to_json(triplets, "triplets.json", graph=graph)

# GraphML (可视化)
export_graph_to_graphml(graph, "graph.graphml")

# RDF Turtle (语义网)
export_graph_to_rdf_turtle(triplets, "graph.ttl")
```

### 任务 5：计算空间相邻关系

```python
from DGGS import spatial_adjacency_to_triplets
from DGGS import Discretizer

grid = Discretizer()
adjacency_triplets = spatial_adjacency_to_triplets(
    cell_tokens=["89c25a3", "89c25a1"],
    grid=grid
)
```

### 任务 6：分析时间变化（多年作物轮作）

```python
from DGGS import temporal_triplets

temporal_data = {
    2020: {"dominant_crop": 1},
    2021: {"dominant_crop": 24},
    2022: {"dominant_crop": 1}
}

rotation_triplets = temporal_triplets("89c25a3", temporal_data)
```

---

## 数据模型速查

### SpatialEntity（空间实体）

```python
from DGGS import SpatialEntity

entity = SpatialEntity(
    entity_id="cdl_89c25a3",
    entity_type="CDLCell",
    attributes={"dggs_level": 12}
)
triplets = entity.to_triplets()
```

### SpatialRelationship（空间关系）

```python
from DGGS import SpatialRelationship

relation = SpatialRelationship(
    subject_id="cdl_89c25a3",
    predicate="contains_crop",
    object_id="crop_corn",
    attributes={"percentage": 86.2}
)
triplets = relation.to_triplets()
```

---

## 三元组格式

### RDF 三元组结构

```
(主体, 谓词, 宾体)
```

### 示例

| 主体 | 谓词 | 宾体 |
|------|------|------|
| cdl_89c25a3 | rdf:type | CDLCell |
| cdl_89c25a3 | attr:dggs_level | 12 |
| cdl_89c25a3 | has_dominant_crop | crop_1 |
| crop_1 | rdf:type | CropType |
| crop_1 | name | Corn |

### GraphReasoning CSV 格式

```
node_1|edge|node_2
cdl_89c25a3|rdf:type|CDLCell
cdl_89c25a3|attr:dggs_level|12
crop_1|name|Corn
```

---

## 函数列表

### 三元组生成（5 个函数）

| 函数 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `discretized_cdl_to_triplets()` | CDL → 三元组 | cell_token, cdl_data | 列表 |
| `discretized_ssurgo_to_triplets()` | SSURGO → 三元组 | cell_token, ssurgo_data | 列表 |
| `discretized_agricultural_intensity_to_triplets()` | 农业强度 → 三元组 | cell_token, intensity_data | 列表 |
| `spatial_adjacency_to_triplets()` | 空间相邻 → 三元组 | cell_tokens, grid | 列表 |
| `temporal_triplets()` | 时间序列 → 三元组 | cell_token, temporal_data | 列表 |

### 图谱操作（3 个函数）

| 函数 | 用途 |
|------|------|
| `create_knowledge_graph_from_discretized_data()` | 从离散化数据创建 NetworkX 图 |
| `merge_into_existing_graph()` | 合并多个图 |
| `triplets_to_dataframe()` | 三元组转 Pandas DataFrame |

### 导出（4 个函数）

| 函数 | 输出格式 | 用途 |
|------|----------|------|
| `export_triplets_to_csv()` | CSV | GraphReasoning 框架 |
| `export_triplets_to_json()` | JSON | 编程使用 |
| `export_graph_to_graphml()` | GraphML | 可视化（Cytoscape/Gephi） |
| `export_graph_to_rdf_turtle()` | RDF Turtle | 语义网应用 |

### 集成（1 个函数）

| 函数 | 用途 |
|------|------|
| `prepare_for_graph_reasoning()` | 一次调用完成整个流程 |

---

## 数据流

```
原始数据（CDL 栅格、SSURGO Shapefile）
        ↓
    [离散化]
        ↓
离散化数据（DGGS 单元格 → 属性字典）
        ↓
    [转换为三元组]
        ↓
RDF 三元组（主-谓-宾）
        ↓
    [构建知识图谱]
        ↓
    NetworkX 有向图
        ↓
    [导出格式]
        ├→ CSV（GraphReasoning）
        ├→ GraphML（可视化）
        ├→ RDF Turtle（语义网）
        └→ JSON（编程）
        ↓
    [与 GraphReasoning 集成]
        ↓
    知识图谱推理和分析
```

---

## 参数速查

### `prepare_for_graph_reasoning()`

```python
prepare_for_graph_reasoning(
    discretized_data,      # Dict[cell_token, data]
    data_type="cdl",       # "cdl", "ssurgo", "intensity", "combined"
    output_dir="./kg"      # 输出目录
)
```

### `create_knowledge_graph_from_discretized_data()`

```python
create_knowledge_graph_from_discretized_data(
    discretized_data,      # Dict[cell_token, data]
    data_type="cdl"        # "cdl", "ssurgo", "intensity", "combined"
)
# 返回: (NetworkX.DiGraph, List[triplets])
```

### `merge_into_existing_graph()`

```python
merge_into_existing_graph(
    existing_graph,        # NetworkX.DiGraph
    new_triplets,          # List[triplets]
    merge_strategy="union" # "union", "intersection", "source_priority"
)
```

### `export_triplets_to_csv()`

```python
export_triplets_to_csv(
    triplets,              # List[(subject, predicate, object)]
    output_file,           # 文件路径
    delimiter="|"          # 分隔符（GraphReasoning 用"|"）
)
```

---

## 示例代码片段

### 完整工作流

```python
from DGGS import (
    discretize_cdl_crop_distribution,
    prepare_for_graph_reasoning
)

# 1. 加载并离散化数据
cdl_pixels = load_cdl_raster("cdl_2021.tif")
cdl_disc = discretize_cdl_crop_distribution(cdl_pixels, level=12)

# 2. 转换并导出
triplets, graph = prepare_for_graph_reasoning(
    discretized_data=cdl_disc,
    data_type="cdl",
    output_dir="./kg"
)

# 3. 与 GraphReasoning 集成
from GraphReasoning.graph_generation import make_graph_from_text
import pandas as pd

df = pd.read_csv("./kg/cdl_triplets.csv", sep="|")
G = make_graph_from_text(df.values.tolist())

print(f"✓ 知识图谱已准备好")
print(f"  节点: {G.number_of_nodes()}")
print(f"  边: {G.number_of_edges()}")
```

### 自定义三元组

```python
from DGGS import SpatialEntity, SpatialRelationship

# 创建自定义实体
entity = SpatialEntity(
    entity_id="myentity_1",
    entity_type="CustomType",
    attributes={"custom_field": "value"}
)

# 转换为三元组
triplets = entity.to_triplets()

# 也可以手动创建
custom_triplets = [
    ("subject1", "predicate1", "object1"),
    ("subject2", "predicate2", "object2")
]
```

---

## 调试技巧

### 检查三元组生成

```python
from examples.raster_examples import discretized_cdl_to_triplets

cell = "89c25a3"
data = {...}
triplets = discretized_cdl_to_triplets(cell, data)

print(f"三元组数量: {len(triplets)}")
for s, p, o in triplets[:10]:
    print(f"  {s} --[{p}]--> {o}")
```

### 验证图谱质量

```python
import networkx as nx

# 检查连通性
is_connected = nx.is_strongly_connected(graph)
print(f"强连通: {is_connected}")

# 检查度数分布
degrees = dict(graph.degree())
print(f"平均度数: {sum(degrees.values()) / len(degrees):.2f}")

# 查找孤立节点
isolates = list(nx.isolates(graph))
print(f"孤立节点数: {len(isolates)}")
```

### 性能监控

```python
import time

start = time.time()
triplets, graph = prepare_for_graph_reasoning(cdl_disc, "cdl")
elapsed = time.time() - start

print(f"处理时间: {elapsed:.2f}s")
print(f"吞吐量: {len(cdl_disc) / elapsed:.1f} 个单元格/秒")
```

---

## 常见问题

**Q: 需要多少三元组？**
A: 通常每个 DGGS 单元格 10-20 个三元组。CDL 数据通常 10-15 个，SSURGO 通常 8-12 个。

**Q: 图谱可以有多大？**
A: NetworkX 可以处理数百万个节点。CSV 通常是瓶颈（文件大小）。

**Q: 如何优化性能？**
A: 使用批处理、选择性导出（仅重要关系）、分层构建小图再合并。

**Q: 支持哪些空间关系？**
A: 包含、相邻、重叠、包含于。详见 DGGS_SPATIAL_RELATIONS_GUIDE.md。

**Q: 可以添加自定义三元组吗？**
A: 可以。只需按 (subject, predicate, object) 格式添加到列表。

---

## 相关资源

- [完整指南](DISCRETIZED_TO_KG_GUIDE.md)
- [CDL 离散化](CDL_DISCRETIZATION_GUIDE.md)
- [SSURGO 离散化](SSURGO_DISCRETIZATION_GUIDE.md)
- [示例代码](examples/discretized_to_kg_examples.py)
- [GraphReasoning 文档](GraphReasoning/README.md)

---

## 获取帮助

查看函数文档：
```python
from DGGS import discretized_to_kg
help(discretized_to_kg.prepare_for_graph_reasoning)
```

运行示例：
```bash
python examples/discretized_to_kg_examples.py
```

检查输出：
```bash
ls -lh kg_output/  # 查看导出的文件
```
