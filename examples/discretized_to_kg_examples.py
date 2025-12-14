"""
从离散化数据到知识图谱的完整工作流示例

演示如何将离散化的地理空间数据 (CDL、SSURGO) 转换成知识图谱，
并集成到 GraphReasoning 框架中。
"""

from pathlib import Path
import sys

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Dggs import (
    discretize_polygon_attributes,
)
from examples.polygon_examples import (
    discretize_ssurgo_soil_properties,
    create_ssurgo_sample_data,
    discretized_ssurgo_to_triplets,
)
from examples.raster_examples import (
    discretize_cdl_crop_distribution,
    discretize_cdl_agricultural_intensity,
    create_cdl_sample_data,
    discretized_cdl_to_triplets,
    discretized_agricultural_intensity_to_triplets,
)
from Dggs.discretized_to_kg import (
    create_knowledge_graph_from_discretized_data,
    triplets_to_dataframe,
    export_triplets_to_csv,
    export_graph_to_graphml,
    export_graph_to_rdf_turtle,
    prepare_for_graph_reasoning,
)
import pandas as pd
import networkx as nx
from pathlib import Path


def example_1_basic_cdl_to_triplets():
    """示例1: 基础的 CDL 数据到三元组转换"""
    print("\n" + "="*70)
    print("示例1: CDL 数据到 RDF 三元组")
    print("="*70)
    
    # 获取离散化的 CDL 数据
    pixels = create_cdl_sample_data(year=2021)
    cdl_result = discretize_cdl_crop_distribution(pixels, level=12)
    
    print(f"✅ 离散化 CDL 数据: {len(cdl_result)} 个单元格\n")
    
    # 转换为三元组
    all_triplets = []
    for cell_token, cell_data in cdl_result.items():
        triplets = discretized_cdl_to_triplets(cell_token, cell_data)
        all_triplets.extend(triplets)
        
        print(f"单元格 {cell_token}:")
        for s, p, o in triplets[:5]:  # 显示前5个三元组
            print(f"  {s} --[{p}]--> {o}")
        if len(triplets) > 5:
            print(f"  ... 及其他 {len(triplets) - 5} 个三元组")
        print()
    
    print(f"📊 总计 {len(all_triplets)} 个 RDF 三元组")
    return all_triplets


def example_2_ssurgo_to_triplets():
    """示例2: SSURGO 土壤数据到三元组转换"""
    print("\n" + "="*70)
    print("示例2: SSURGO 土壤数据到 RDF 三元组")
    print("="*70)
    
    # 获取离散化的 SSURGO 数据
    map_units = create_ssurgo_sample_data()
    ssurgo_result = discretize_ssurgo_soil_properties(
        map_units,
        properties=['pH', 'sand_percent', 'clay_percent', 'bulk_density'],
        level=12
    )
    
    print(f"✅ 离散化 SSURGO 数据: {len(ssurgo_result)} 个单元格\n")
    
    # 转换为三元组
    all_triplets = []
    for cell_token, cell_data in ssurgo_result.items():
        triplets = discretized_ssurgo_to_triplets(cell_token, cell_data)
        all_triplets.extend(triplets)
        
        print(f"单元格 {cell_token}:")
        for s, p, o in triplets[:4]:
            print(f"  {s} --[{p}]--> {o}")
        if len(triplets) > 4:
            print(f"  ... 及其他 {len(triplets) - 4} 个三元组")
        print()
    
    print(f"📊 总计 {len(all_triplets)} 个 RDF 三元组")
    return all_triplets


def example_3_intensity_to_triplets():
    """示例3: 农业强度评估到三元组转换"""
    print("\n" + "="*70)
    print("示例3: 农业强度到 RDF 三元组")
    print("="*70)
    
    pixels = create_cdl_sample_data(year=2021)
    intensity_result = discretize_cdl_agricultural_intensity(pixels, level=12)
    
    print(f"✅ 强度评估: {len(intensity_result)} 个单元格\n")
    
    all_triplets = []
    for cell_token, cell_data in intensity_result.items():
        triplets = discretized_agricultural_intensity_to_triplets(cell_token, cell_data)
        all_triplets.extend(triplets)
        
        print(f"单元格 {cell_token}:")
        print(f"  强度等级: {cell_data['intensity']}")
        print(f"  强度分数: {cell_data['intensity_score']:.1f}")
        print(f"  单一作物: {cell_data['monoculture']}")
        print(f"  三元组数: {len(triplets)}")
        print()
    
    print(f"📊 总计 {len(all_triplets)} 个 RDF 三元组")
    return all_triplets


def example_4_create_knowledge_graph():
    """示例4: 从离散化数据创建知识图谱"""
    print("\n" + "="*70)
    print("示例4: 创建知识图谱 (NetworkX)")
    print("="*70)
    
    # 获取离散化数据
    pixels = create_cdl_sample_data(year=2021)
    cdl_result = discretize_cdl_crop_distribution(pixels, level=12)
    
    # 创建知识图谱
    G, triplets_list = create_knowledge_graph_from_discretized_data(
        cdl_result,
        triplet_converter=discretized_cdl_to_triplets,
        include_spatial=False  # 暂不包括空间关系
    )
    
    print(f"✅ 知识图谱创建成功\n")
    print(f"📊 图谱统计:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  平均度数: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  是否连通: {nx.is_connected(G.to_undirected())}")
    
    # 显示图谱结构
    print(f"\n📍 节点示例 (前10个):")
    for i, node in enumerate(list(G.nodes())[:10]):
        neighbors = list(G.neighbors(node))[:3]
        print(f"  {node} -> {neighbors}")
    
    print(f"\n🔗 边的类型:")
    edge_types = {}
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', 'unknown')
        edge_types[rel] = edge_types.get(rel, 0) + 1
    
    for rel, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {rel}: {count}")
    
    return G, triplets_list


def example_5_export_formats():
    """示例5: 导出多种格式"""
    print("\n" + "="*70)
    print("示例5: 导出知识图谱到多种格式")
    print("="*70)
    
    # 生成数据
    pixels = create_cdl_sample_data(year=2021)
    cdl_result = discretize_cdl_crop_distribution(pixels, level=12)
    
    # 转换为三元组
    all_triplets = []
    for cell_token, cell_data in cdl_result.items():
        all_triplets.extend(discretized_cdl_to_triplets(cell_token, cell_data))
    
    # 创建输出目录（统一放在 output/ 下）
    output_dir = Path('./output/kg_export_example/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出 CSV 格式 (适用于 GraphReasoning)
    csv_path = output_dir / 'triplets.csv'
    export_triplets_to_csv(all_triplets, str(csv_path))
    print(f"✅ CSV 导出: {csv_path}")
    
    # 导出 GraphML 格式 (NetworkX)
    G = nx.DiGraph()
    for s, p, o in all_triplets:
        G.add_edge(s, o, relation=p, label=p)
    
    graphml_path = output_dir / 'graph.graphml'
    export_graph_to_graphml(G, str(graphml_path))
    print(f"✅ GraphML 导出: {graphml_path}")
    
    # 导出 RDF Turtle 格式
    turtle_path = output_dir / 'graph.ttl'
    export_graph_to_rdf_turtle(all_triplets, str(turtle_path))
    print(f"✅ RDF Turtle 导出: {turtle_path}")
    
    # 显示 CSV 内容
    print(f"\n📄 CSV 内容预览:")
    df = pd.read_csv(csv_path, sep='|')
    print(df.head(10).to_string())
    
    return output_dir


def example_6_prepare_for_graphreasoning():
    """示例6: 准备数据用于 GraphReasoning 框架"""
    print("\n" + "="*70)
    print("示例6: 为 GraphReasoning 准备数据")
    print("="*70)
    
    # 获取离散化数据
    pixels = create_cdl_sample_data(year=2021)
    cdl_result = discretize_cdl_crop_distribution(pixels, level=12)
    
    # 准备数据
    triplets, G = prepare_for_graph_reasoning(
        cdl_result,
        triplet_converter=discretized_cdl_to_triplets,
        data_type='cdl',  # 用于输出文件命名
        output_dir='./output/kg_for_graphreasoning/'
    )
    
    print(f"\n💡 下一步: 使用 GraphReasoning 框架进行推理")
    print(f"""
from GraphConstruct.graph_generation import make_graph_from_text
from Llms.llm_providers import get_generate_fn

# 1. 设置 LLM 提供器
provider_config = {{"model": "gpt-4", "api_key": "your_key"}}
generate = get_generate_fn("openai", provider_config)

# 2. 从 CSV 三元组创建图
df = pd.read_csv('./output/kg_for_graphreasoning/cdl_triplets.csv', sep='|')

# 3. 用于推理
from GraphReasoning.graph_analysis import find_path_and_reason
result = find_path_and_reason(
    G,
    keyword_1="corn",
    keyword_2="intensive agriculture",
    generate=generate
)
    """)


def example_7_integrated_workflow():
    """示例7: 完整的集成工作流（CDL + SSURGO）"""
    print("\n" + "="*70)
    print("示例7: 完整集成工作流 (CDL + SSURGO)")
    print("="*70)
    
    # 第1步: 离散化 CDL 数据
    print("\n📍 步骤1: 离散化 CDL 数据...")
    cdl_pixels = create_cdl_sample_data(year=2021)
    cdl_result = discretize_cdl_crop_distribution(cdl_pixels, level=12)
    print(f"✅ 获得 {len(cdl_result)} 个 CDL 单元格")
    
    # 第2步: 离散化 SSURGO 数据
    print("\n📍 步骤2: 离散化 SSURGO 数据...")
    map_units = create_ssurgo_sample_data()
    ssurgo_result = discretize_ssurgo_soil_properties(
        map_units, 
        properties=['pH', 'sand_percent', 'clay_percent'],
        level=12
    )
    print(f"✅ 获得 {len(ssurgo_result)} 个土壤单元格")
    
    # 第3步: 转换为三元组
    print("\n📍 步骤3: 转换为 RDF 三元组...")
    cdl_triplets = []
    for cell, data in cdl_result.items():
        cdl_triplets.extend(discretized_cdl_to_triplets(cell, data))
    
    ssurgo_triplets = []
    for cell, data in ssurgo_result.items():
        ssurgo_triplets.extend(discretized_ssurgo_to_triplets(cell, data))
    
    print(f"✅ CDL 三元组: {len(cdl_triplets)}")
    print(f"✅ SSURGO 三元组: {len(ssurgo_triplets)}")
    
    # 第4步: 创建知识图谱
    print("\n📍 步骤4: 创建知识图谱...")
    G = nx.DiGraph()
    
    # 添加所有三元组
    for s, p, o in cdl_triplets + ssurgo_triplets:
        G.add_edge(s, o, relation=p, label=p)
    
    print(f"✅ 知识图谱创建:")
    print(f"   - 节点: {G.number_of_nodes()}")
    print(f"   - 边: {G.number_of_edges()}")
    
    # 第5步: 创建跨越两个数据源的关系
    print("\n📍 步骤5: 创建跨数据源关系...")
    cross_domain_edges = 0
    
    for cell_token in cdl_result.keys():
        if cell_token in ssurgo_result:
            # 添加 CDL 单元格到 SSURGO 单元格的关系
            G.add_edge(
                f'cdl_{cell_token}',
                f'soil_{cell_token}',
                relation='spatially_coincident_with',
                label='spatially_coincident_with'
            )
            cross_domain_edges += 1
    
    print(f"✅ 添加 {cross_domain_edges} 个跨域关系")
    
    # 第6步: 分析图谱
    print("\n📍 步骤6: 分析知识图谱...")
    print(f"✅ 连通分量数: {nx.number_connected_components(G.to_undirected())}")
    print(f"✅ 平均最短路径长度: {nx.average_shortest_path_length(G.to_undirected()):.2f}")
    
    # 查找重要节点
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n✅ 最重要节点 (按度数):")
    for node, degree in top_nodes:
        print(f"   - {node}: {degree} 条连接")
    
    # 第7步: 导出
    print("\n📍 步骤7: 导出知识图谱...")
    output_dir = Path('./output/kg_integrated/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    export_graph_to_graphml(G, str(output_dir / 'integrated_kg.graphml'))
    print(f"✅ 已导出到 {output_dir / 'integrated_kg.graphml'}")
    
    return G


if __name__ == "__main__":
    print("\n" + "🌍 离散化数据到知识图谱转换示例")
    print("="*70)
    print("演示如何将地理空间离散化数据转换成知识图谱")
    
    # 运行所有示例
    example_1_basic_cdl_to_triplets()
    example_2_ssurgo_to_triplets()
    example_3_intensity_to_triplets()
    example_4_create_knowledge_graph()
    example_5_export_formats()
    example_6_prepare_for_graphreasoning()
    example_7_integrated_workflow()
    
    print("\n" + "="*70)
    print("✅ 所有示例完成！")
    print("="*70 + "\n")
