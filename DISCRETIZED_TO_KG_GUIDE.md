# 离散化数据到知识图谱转换指南

## 概述

本指南展示如何将离散化的地理空间数据（CDL 作物数据、SSURGO 土壤数据）转换为知识图谱。该流程支持：

- **RDF 三元组生成**：将地理空间数据转换为标准的主谓宾格式
- **图谱构建**：使用 NetworkX 创建知识图谱
- **多格式导出**：CSV (GraphReasoning)、GraphML (可视化)、RDF Turtle (语义网)、JSON (编程)
- **图谱集成**：与 GraphReasoning 框架无缝集成

---

## 核心模块：`DGGS/discretized_to_kg.py`

### 数据模型

#### SpatialEntity（空间实体）

表示 DGGS 单元格中的一个地理空间对象。

```python
from DGGS import SpatialEntity

entity = SpatialEntity(
    entity_id="cdl_89c25a3",           # 唯一标识符
    entity_type="CDLCell",              # 实体类型
    attributes={
        "dggs_level": 12,
        "total_pixels": 58,
        "total_area_acres": 11.61
    }
)

# 转换为 RDF 三元组
triplets = entity.to_triplets()
# [
#   ("cdl_89c25a3", "rdf:type", "CDLCell"),
#   ("cdl_89c25a3", "attr:dggs_level", "12"),
#   ("cdl_89c25a3", "attr:total_pixels", "58"),
#   ("cdl_89c25a3", "attr:total_area_acres", "11.61")
# ]
```

#### SpatialRelationship（空间关系）

表示两个实体之间的关系。

```python
from DGGS import SpatialRelationship

relation = SpatialRelationship(
    subject_id="cdl_89c25a3",
    predicate="contains_crop",
    object_id="crop_corn",
    attributes={"percentage": 86.2, "area_acres": 10.01}
)

triplets = relation.to_triplets()
# [
#   ("cdl_89c25a3", "contains_crop", "crop_corn"),
#   ("crop_corn", "percentage", "86.2"),
#   ("crop_corn", "area_acres", "10.01")
# ]
```

---

## 三元组生成函数

### 1. CDL 数据 → 三元组

```python
# CDL 转三元组函数现在在 examples/raster_examples.py 中
from examples.raster_examples import discretized_cdl_to_triplets

# 输入：离散化的 CDL 单元格数据
cell_token = "89c25a3"
cdl_data = {
    "dggs_level": 12,
    "total_pixels": 58,
    "total_area_acres": 11.61,
    "dominant_crop": 1,  # Corn
    "crop_distribution": {1: 0.862, 24: 0.138},
    "crop_names": {1: "Corn", 24: "Soybeans"}
}

# 转换为三元组
triplets = discretized_cdl_to_triplets(cell_token, cdl_data)

# 输出：RDF 三元组列表
# [
#   ("cdl_89c25a3", "rdf:type", "CDLCell"),
#   ("cdl_89c25a3", "attr:dggs_level", "12"),
#   ("cdl_89c25a3", "has_dominant_crop", "crop_1"),
#   ("crop_1", "rdf:type", "CropType"),
#   ("crop_1", "name", "Corn"),
#   ("cdl_89c25a3", "contains_crop", "crop_corn"),
#   ("crop_corn", "percentage", "86.2"),
#   ("crop_corn", "area_acres", "10.01"),
#   ...
# ]
```

### 2. SSURGO 数据 → 三元组

```python
# SSURGO 转三元组函数现在在 examples/polygon_examples.py 中
from examples.polygon_examples import discretized_ssurgo_to_triplets

cell_token = "89c25a3"
ssurgo_data = {
    "dggs_level": 12,
    "source": "SSURGO",
    "soil_properties": {
        "pH": {"value": 6.2, "method": "weighted_mean"},
        "sand_percent": {"value": 35.0, "method": "weighted_mean"},
        "clay_percent": {"value": 28.5, "method": "weighted_mean"}
    },
    "soil_series": "Miami"
}

# 转换为三元组
triplets = discretized_ssurgo_to_triplets(cell_token, ssurgo_data)

# 输出：土壤属性的 RDF 三元组
# [
#   ("soil_89c25a3", "rdf:type", "SoilCell"),
#   ("soil_89c25a3", "soil_property", "pH:weighted:mean=6.2"),
#   ("soil_89c25a3", "soil_property", "sand_percent:weighted:mean=35.0"),
#   ...
# ]
```

### 3. 农业强度 → 三元组

```python
from DGGS import discretized_agricultural_intensity_to_triplets

cell_token = "89c25a3"
intensity_data = {
    "intensity_score": 100.0,
    "intensity_level": "intensive",
    "single_crop": True,
    "crop_diversity": 1
}

triplets = discretized_agricultural_intensity_to_triplets(cell_token, intensity_data)

# 输出：强度分类的三元组
# [
#   ("intensity_89c25a3", "rdf:type", "AgriculturalIntensity"),
#   ("intensity_89c25a3", "intensity_score", "100.0"),
#   ("intensity_89c25a3", "intensity_level", "intensive"),
#   ...
# ]
```

### 4. 空间相邻关系 → 三元组

```python
from DGGS import spatial_adjacency_to_triplets

# 获取 DGGS 单元格的相邻单元格
cell_tokens = ["89c25a3", "89c25a1", "89c259f"]
grid = dggs.Discretizer()

# 计算相邻关系
triplets = spatial_adjacency_to_triplets(cell_tokens, grid)

# 输出：相邻关系的三元组
# [
#   ("89c25a3", "adjacent_to", "89c25a1"),
#   ("89c25a1", "adjacent_to", "89c259f"),
#   ...
# ]
```

### 5. 时间关系 → 三元组

```python
from DGGS import temporal_triplets

cell_token = "89c25a3"
temporal_data = {
    2020: {"dominant_crop": 1, "crop_area": 11.2},  # Corn
    2021: {"dominant_crop": 24, "crop_area": 11.6}, # Soybeans
    2022: {"dominant_crop": 1, "crop_area": 10.9}   # Corn
}

# 生成时间关系
triplets = temporal_triplets(cell_token, temporal_data)

# 输出：作物轮作模式
# [
#   ("cdl_89c25a3_2020", "rdf:type", "CDLCell"),
#   ("cdl_89c25a3_2020", "temporal:year", "2020"),
#   ("cdl_89c25a3_2020", "crop", "corn"),
#   ("cdl_89c25a3_2021", "crop", "soybeans"),
#   ("cdl_89c25a3_2020", "temporal:next_year", "cdl_89c25a3_2021"),
#   ...
# ]
```

---

## 知识图谱构建

### 从三元组创建 NetworkX 图

```python
from DGGS import create_knowledge_graph_from_discretized_data
import pandas as pd

# 示例 1：从 CDL 数据创建图
cdl_data = {
    "89c25a3": {"dggs_level": 12, "dominant_crop": 1, ...},
    "89c25a1": {"dggs_level": 12, "dominant_crop": 5, ...}
}

graph, triplets = create_knowledge_graph_from_discretized_data(
    discretized_data=cdl_data,
    data_type="cdl"
)

print(f"节点数: {graph.number_of_nodes()}")
print(f"边数: {graph.number_of_edges()}")

# 示例 2：从 SSURGO 数据创建图
ssurgo_data = {
    "89c25a3": {"soil_series": "Miami", "pH": 6.2, ...}
}

soil_graph, soil_triplets = create_knowledge_graph_from_discretized_data(
    discretized_data=ssurgo_data,
    data_type="ssurgo"
)
```

### 合并多个图

```python
from DGGS import merge_into_existing_graph

# 将 SSURGO 图合并到 CDL 图中
merged_graph = merge_into_existing_graph(
    existing_graph=graph,
    new_triplets=soil_triplets,
    merge_strategy="union"  # 或 "intersection", "source_priority"
)

print(f"合并后 - 节点数: {merged_graph.number_of_nodes()}")
print(f"合并后 - 边数: {merged_graph.number_of_edges()}")
```

---

## 导出知识图谱

### 导出为 CSV（GraphReasoning 兼容）

```python
from DGGS import export_triplets_to_csv
import pandas as pd

# 导出为 CSV
output_file = "kg_triplets.csv"
export_triplets_to_csv(triplets, output_file)

# 验证输出格式
df = pd.read_csv(output_file, sep="|")
print(df.head())
#         node_1          edge      node_2
# 0  cdl_89c25a3      rdf:type      CDLCell
# 1  cdl_89c25a3  attr:dggs_level         12
# 2  cdl_89c25a3  attr:total_pixels         58
```

### 导出为 GraphML（可视化）

```python
from DGGS import export_graph_to_graphml

# 导出为 GraphML
output_file = "kg.graphml"
export_graph_to_graphml(graph, output_file)

# 在 Cytoscape 或 Gephi 中打开进行可视化
```

### 导出为 RDF Turtle（语义网）

```python
from DGGS import export_graph_to_rdf_turtle

# 导出为 RDF Turtle 格式
output_file = "kg.ttl"
export_graph_to_rdf_turtle(triplets, output_file, base_uri="http://example.org/geokg/")

# 验证输出
with open(output_file, "r") as f:
    print(f.read()[:500])
```

### 导出为 JSON

```python
from DGGS import export_triplets_to_json

# 导出为 JSON
output_file = "kg.json"
export_triplets_to_json(triplets, output_file, graph=graph)
```

---

## 与 GraphReasoning 框架集成

### 快速集成

```python
from DGGS import prepare_for_graph_reasoning

# 一次调用完成所有步骤
triplets, graph = prepare_for_graph_reasoning(
    discretized_data=cdl_data,
    data_type="cdl",
    output_dir="./output/kg_output"
)

# 生成的文件：
# - output/kg_output/cdl_triplets.csv    (GraphReasoning 格式)
# - output/kg_output/cdl_graph.graphml   (可视化)
# - output/kg_output/...                 (其他格式)
```

### 与 GraphReasoning 推理引擎集成

```python
from GraphReasoning.graph_generation import make_graph_from_text
from GraphReasoning.graph_analysis import find_path_and_reason
from GraphReasoning.llm_providers import get_generate_fn
import pandas as pd

# 1. 加载三元组
df = pd.read_csv("kg_output/cdl_triplets.csv", sep="|")

# 2. 创建 NetworkX 图
from GraphReasoning.graph_generation import make_graph_from_text
G = make_graph_from_text(df.values.tolist())

# 3. 初始化 LLM
provider_config = {"model": "gpt-4", "api_key": "your_key"}
generate = get_generate_fn("openai", provider_config)

# 4. 图谱推理
result = find_path_and_reason(
    G,
    keyword_1="corn",
    keyword_2="intensive_agriculture",
    generate=generate
)

print(f"推理结果:\n{result}")
```

---

## 完整工作流示例

### 场景：从原始数据到知识图谱推理

```python
# 领域专用函数在各自的示例文件中
from examples.raster_examples import (
    discretize_cdl_crop_distribution,
    discretized_cdl_to_triplets,
)
from examples.polygon_examples import (
    discretize_ssurgo_soil_properties,
    discretized_ssurgo_to_triplets,
)
# 通用知识图谱工具在 DGGS 包中
from DGGS import (
    create_knowledge_graph_from_discretized_data,
    merge_into_existing_graph,
    prepare_for_graph_reasoning
)

# 步骤 1：加载并离散化原始数据
print("步骤 1: 离散化原始数据...")

# CDL 作物数据
cdl_pixels = load_cdl_raster("cdl_2021.tif")
cdl_discretized = discretize_cdl_crop_distribution(cdl_pixels, level=12)
print(f"✓ 离散化了 {len(cdl_discretized)} 个 CDL 单元格")

# SSURGO 土壤数据
soil_map_units = load_ssurgo_shapefile("soils.shp")
soil_discretized = discretize_ssurgo_soil_properties(
    soil_map_units,
    properties=["pH", "sand_percent", "clay_percent"],
    level=12
)
print(f"✓ 离散化了 {len(soil_discretized)} 个土壤单元格")

# 步骤 2：转换为三元组
print("\n步骤 2: 转换为 RDF 三元组...")
cdl_triplets = []
for cell, data in cdl_discretized.items():
    cdl_triplets.extend(discretized_cdl_to_triplets(cell, data))

soil_triplets = []
for cell, data in soil_discretized.items():
    soil_triplets.extend(discretized_ssurgo_to_triplets(cell, data))

print(f"✓ 生成了 {len(cdl_triplets)} 个 CDL 三元组")
print(f"✓ 生成了 {len(soil_triplets)} 个土壤三元组")

# 步骤 3：构建知识图谱
print("\n步骤 3: 构建知识图谱...")
cdl_graph, _ = create_knowledge_graph_from_discretized_data(
    cdl_discretized, 
    data_type="cdl"
)

soil_graph, _ = create_knowledge_graph_from_discretized_data(
    soil_discretized,
    data_type="ssurgo"
)

# 合并图
combined_graph = merge_into_existing_graph(
    existing_graph=cdl_graph,
    new_triplets=soil_triplets,
    merge_strategy="union"
)
print(f"✓ 合并后: 节点 {combined_graph.number_of_nodes()}, 边 {combined_graph.number_of_edges()}")

# 步骤 4：导出为 GraphReasoning 格式
print("\n步骤 4: 准备用于推理...")
triplets, graph = prepare_for_graph_reasoning(
    discretized_data=cdl_discretized,
    data_type="cdl",
    output_dir="./output/kg_final"
)

print(f"✓ 已导出到 ./output/kg_final/")
print(f"  - 三元组: {len(triplets)}")
print(f"  - 图谱节点: {graph.number_of_nodes()}")
print(f"  - 图谱边: {graph.number_of_edges()}")

# 步骤 5：使用 GraphReasoning 进行推理
print("\n步骤 5: 知识图谱推理...")
# (见上面的集成示例)
```

---

## 性能优化建议

### 大规模数据处理

对于大型数据集（>100K 个 DGGS 单元格），考虑以下优化：

```python
# 1. 批量处理三元组生成
batch_size = 1000
triplets = []

for i in range(0, len(discretized_data), batch_size):
    batch = dict(list(discretized_data.items())[i:i+batch_size])
    for cell, data in batch.items():
        triplets.extend(discretized_cdl_to_triplets(cell, data))

# 2. 使用数据框进行高效导出
import pandas as pd

df = pd.DataFrame(
    triplets,
    columns=["node_1", "edge", "node_2"]
)
df.to_csv("triplets.csv", sep="|", index=False)

# 3. 分层图构建
# 构建较小的子图，然后合并
sub_graphs = []
for chunk in chunks(discretized_data, 10000):
    g, _ = create_knowledge_graph_from_discretized_data(chunk, "cdl")
    sub_graphs.append(g)

# 使用 NetworkX 高效合并
import networkx as nx
final_graph = nx.compose_all(sub_graphs)
```

---

## 故障排除

### 问题 1：三元组数量意外

**症状**：生成的三元组数远少于预期

**解决**：
```python
# 检查数据完整性
for cell, data in discretized_data.items():
    triplets = discretized_cdl_to_triplets(cell, data)
    if len(triplets) < 5:
        print(f"警告: {cell} 只有 {len(triplets)} 个三元组")
        print(f"数据: {data}")
```

### 问题 2：GraphML 文件太大

**症状**：GraphML 文件超过几 GB

**解决**：
```python
# 只导出重要的关系类型
important_triplets = [
    t for t in triplets 
    if t[1] in ["rdf:type", "has_dominant_crop", "soil_property"]
]

export_graph_to_graphml(
    create_knowledge_graph_from_discretized_data(
        important_triplets
    ),
    "slim_kg.graphml"
)
```

### 问题 3：与 GraphReasoning 集成失败

**症状**：CSV 导入到 GraphReasoning 时出错

**解决**：
```python
# 验证 CSV 格式
import pandas as pd

df = pd.read_csv("triplets.csv", sep="|")
assert list(df.columns) == ["node_1", "edge", "node_2"]
assert df.shape[0] > 0
assert not df.isna().any().any()

print("✓ CSV 格式验证通过")
```

---

## 相关文档

- [CDL 离散化指南](CDL_DISCRETIZATION_GUIDE.md)
- [SSURGO 离散化指南](SSURGO_DISCRETIZATION_GUIDE.md)
- [DGGS 空间关系指南](DGGS_SPATIAL_RELATIONS_GUIDE.md)
- [GraphReasoning 框架](GraphReasoning/README.md)

---

## API 参考

详见 `DGGS/discretized_to_kg.py` 中的完整文档字符串。

快速开始：
```python
from DGGS import discretized_to_kg
help(discretized_to_kg.discretized_cdl_to_triplets)
help(discretized_to_kg.create_knowledge_graph_from_discretized_data)
help(discretized_to_kg.prepare_for_graph_reasoning)
```
