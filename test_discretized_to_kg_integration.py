#!/usr/bin/env python
"""
完整 Pipeline 测试: 空间数据离散化 → 知识图谱构建
Complete Pipeline Test: Spatial Data Discretization → Knowledge Graph Construction

测试流程:
1. 原始空间数据 (CDL 像素, SSURGO 地图单元)
2. DGGS 离散化 (空间聚合)
3. RDF 三元组生成
4. 知识图谱构建 (NetworkX)
5. 多格式导出
6. GraphReasoning 集成
"""

import sys
import os
from pathlib import Path
import tempfile

def test_imports():
    """测试所有必要的导入"""
    print("=" * 70)
    print("测试 1: 导入验证")
    print("=" * 70)
    
    try:
        from DGGS import (
            # 数据模型
            SpatialEntity,
            SpatialRelationship,
            # 三元组生成 (通用工具)
            discretized_agricultural_intensity_to_triplets,
            spatial_adjacency_to_triplets,
            temporal_triplets,
            # 图谱操作
            create_knowledge_graph_from_discretized_data,
            triplets_to_dataframe,
            merge_into_existing_graph,
            # 导出
            export_triplets_to_csv,
            export_triplets_to_json,
            export_graph_to_graphml,
            export_graph_to_rdf_turtle,
            # 集成
            prepare_for_graph_reasoning
        )
        # 领域专用三元组生成函数现在在示例文件中
        from examples.raster_examples import discretized_cdl_to_triplets
        from examples.polygon_examples import discretized_ssurgo_to_triplets
        
        print("✅ 所有导入成功!")
        print(f"   - SpatialEntity: {SpatialEntity}")
        print(f"   - 三元组生成函数: 5 个")
        print(f"   - 图谱操作函数: 3 个")
        print(f"   - 导出函数: 4 个")
        print(f"   - 集成函数: 1 个")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_spatial_entity():
    """测试 SpatialEntity 类"""
    print("\n" + "=" * 70)
    print("测试 2: SpatialEntity 数据模型")
    print("=" * 70)
    
    try:
        from DGGS import SpatialEntity
        
        entity = SpatialEntity(
            entity_id="test_entity_1",
            entity_type="TestType",
            attributes={"attr1": "value1", "attr2": 123}
        )
        
        triplets = entity.to_triplets()
        
        print(f"✅ SpatialEntity 创建成功")
        print(f"   - 实体 ID: {entity.entity_id}")
        print(f"   - 实体类型: {entity.entity_type}")
        print(f"   - 属性数: {len(entity.attributes)}")
        print(f"   - 生成的三元组数: {len(triplets)}")
        for i, (s, p, o) in enumerate(triplets):
            print(f"     {i+1}. {s} --[{p}]--> {o}")
        
        assert len(triplets) > 0, "应生成至少一个三元组"
        return True
    except Exception as e:
        print(f"❌ SpatialEntity 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spatial_relationship():
    """测试 SpatialRelationship 类"""
    print("\n" + "=" * 70)
    print("测试 3: SpatialRelationship 数据模型")
    print("=" * 70)
    
    try:
        from DGGS import SpatialRelationship
        
        relation = SpatialRelationship(
            source_id="entity_1",
            target_id="entity_2",
            rel_type="test_relation",
            properties={"weight": 0.8}
        )
        
        triplet = relation.to_triplet()
        
        print(f"✅ SpatialRelationship 创建成功")
        print(f"   - 源: {relation.source_id}")
        print(f"   - 关系类型: {relation.rel_type}")
        print(f"   - 目标: {relation.target_id}")
        print(f"   - 生成的三元组: {triplet}")
        
        assert triplet and len(triplet) == 3, "应生成正确格式的三元组"
        return True
    except Exception as e:
        print(f"❌ SpatialRelationship 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_triplet_generation():
    """测试三元组生成 - 使用真实离散化数据"""
    print("\n" + "=" * 70)
    print("测试 4: 完整离散化 → 三元组生成")
    print("=" * 70)
    
    try:
        # 导入离散化函数和三元组转换函数
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
        )
        from DGGS import (
            discretized_agricultural_intensity_to_triplets
        )
        
        # 步骤 1: 生成原始 CDL 数据
        print("  步骤 1/4: 生成原始 CDL 像素数据...")
        cdl_pixels = create_cdl_sample_data(year=2021)
        print(f"    ✓ 生成了 {len(cdl_pixels)} 个 CDL 像素")
        
        # 步骤 2: 离散化到 DGGS 单元格
        print("  步骤 2/4: 离散化到 DGGS 单元格...")
        cdl_discretized = discretize_cdl_crop_distribution(cdl_pixels, level=12)
        print(f"    ✓ 离散化为 {len(cdl_discretized)} 个 DGGS 单元格")
        
        # 步骤 3: 转换为 RDF 三元组
        print("  步骤 3/4: 生成 RDF 三元组...")
        first_cell = list(cdl_discretized.keys())[0]
        first_data = cdl_discretized[first_cell]
        
        print(f"    单元格数据结构: {list(first_data.keys())}")
        print(f"    dominant_crop: {first_data.get('dominant_crop')}")
        
        cdl_triplets = discretized_cdl_to_triplets(first_cell, first_data)
        print(f"    ✓ CDL 三元组生成: {len(cdl_triplets)} 个")
        
        # 测试 SSURGO 工作流
        print("  步骤 4/4: 测试 SSURGO 工作流...")
        ssurgo_map_units = create_ssurgo_sample_data()
        ssurgo_discretized = discretize_ssurgo_soil_properties(
            ssurgo_map_units,
            properties=['pH', 'sand_percent', 'clay_percent'],
            level=12
        )
        
        first_soil_cell = list(ssurgo_discretized.keys())[0]
        first_soil_data = ssurgo_discretized[first_soil_cell]
        
        ssurgo_triplets = discretized_ssurgo_to_triplets(first_soil_cell, first_soil_data)
        print(f"    ✓ SSURGO 三元组生成: {len(ssurgo_triplets)} 个")
        
        # 测试农业强度
        intensity_discretized = discretize_cdl_agricultural_intensity(cdl_pixels, level=12)
        first_intensity_cell = list(intensity_discretized.keys())[0]
        first_intensity_data = intensity_discretized[first_intensity_cell]
        
        intensity_triplets = discretized_agricultural_intensity_to_triplets(
            first_intensity_cell, 
            first_intensity_data
        )
        print(f"    ✓ 农业强度三元组生成: {len(intensity_triplets)} 个")
        
        print(f"\n  ✅ 完整工作流测试通过")
        print(f"     原始像素 → 离散化 → 三元组生成成功")
        
        return True
    except Exception as e:
        print(f"❌ 三元组生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_creation():
    """测试知识图谱创建 - 完整工作流"""
    print("\n" + "=" * 70)
    print("测试 5: 完整知识图谱创建工作流")
    print("=" * 70)
    
    try:
        from DGGS import (
            create_knowledge_graph_from_discretized_data
        )
        from examples.polygon_examples import (
            discretize_ssurgo_soil_properties,
            create_ssurgo_sample_data,
        )
        from examples.raster_examples import (
            discretize_cdl_crop_distribution,
            create_cdl_sample_data,
        )
        
        # 完整 Pipeline: 原始数据 → 离散化 → 知识图谱
        print("  步骤 1/4: 准备原始空间数据...")
        cdl_pixels = create_cdl_sample_data(year=2021)
        ssurgo_map_units = create_ssurgo_sample_data()
        print(f"    ✓ CDL 像素: {len(cdl_pixels)}")
        print(f"    ✓ SSURGO 地图单元: {len(ssurgo_map_units)}")
        
        print("  步骤 2/4: DGGS 空间离散化...")
        cdl_discretized = discretize_cdl_crop_distribution(cdl_pixels, level=12)
        ssurgo_discretized = discretize_ssurgo_soil_properties(
            ssurgo_map_units,
            properties=['pH', 'sand_percent'],
            level=12
        )
        print(f"    ✓ CDL 离散化单元格: {len(cdl_discretized)}")
        print(f"    ✓ SSURGO 离散化单元格: {len(ssurgo_discretized)}")
        
        print("  步骤 3/4: 构建知识图谱...")
        cdl_graph, cdl_triplets = create_knowledge_graph_from_discretized_data(
            cdl_discretized, 
            "cdl"
        )
        ssurgo_graph, ssurgo_triplets = create_knowledge_graph_from_discretized_data(
            ssurgo_discretized,
            "ssurgo"
        )
        
        print(f"    ✓ CDL 图谱 - 节点: {cdl_graph.number_of_nodes()}, 边: {cdl_graph.number_of_edges()}")
        print(f"    ✓ SSURGO 图谱 - 节点: {ssurgo_graph.number_of_nodes()}, 边: {ssurgo_graph.number_of_edges()}")
        
        print("  步骤 4/4: 合并多源知识图谱...")
        from DGGS import merge_into_existing_graph
        merged_graph = merge_into_existing_graph(
            cdl_graph,
            ssurgo_triplets,
            merge_strategy="union"
        )
        print(f"    ✓ 合并后 - 节点: {merged_graph.number_of_nodes()}, 边: {merged_graph.number_of_edges()}")
        
        assert cdl_graph.number_of_nodes() > 0, "CDL图应有节点"
        assert cdl_graph.number_of_edges() > 0, "CDL图应有边"
        assert merged_graph.number_of_nodes() >= cdl_graph.number_of_nodes(), "合并后节点数应增加"
        
        print(f"\n  ✅ 完整知识图谱创建成功")
        print(f"     原始数据 → 离散化 → 三元组 → NetworkX 图 → 合并")
        
        return True
    except Exception as e:
        print(f"❌ 图创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_formats():
    """测试导出格式 - 完整工作流"""
    print("\n" + "=" * 70)
    print("测试 6: 完整 Pipeline → 多格式导出")
    print("=" * 70)
    
    try:
        from DGGS import (
            create_knowledge_graph_from_discretized_data,
            export_triplets_to_csv,
            export_triplets_to_json,
            export_graph_to_graphml,
            export_graph_to_rdf_turtle
        )
        from examples.raster_examples import (
            discretize_cdl_crop_distribution,
            create_cdl_sample_data,
        )
        
        # 完整工作流
        print("  步骤 1/3: 原始数据 → 离散化...")
        cdl_pixels = create_cdl_sample_data(year=2021)
        cdl_discretized = discretize_cdl_crop_distribution(cdl_pixels, level=12)
        print(f"    ✓ 离散化为 {len(cdl_discretized)} 个单元格")
        
        print("  步骤 2/3: 离散化数据 → 知识图谱...")
        graph, triplets = create_knowledge_graph_from_discretized_data(
            cdl_discretized,
            "cdl"
        )
        print(f"    ✓ 图谱: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"    ✓ 三元组: {len(triplets)} 个")
        
        print("  步骤 3/3: 导出到多种格式...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # CSV (GraphReasoning 兼容)
            csv_file = os.path.join(tmpdir, "kg.csv")
            export_triplets_to_csv(triplets, csv_file)
            csv_size = os.path.getsize(csv_file)
            print(f"    ✓ CSV 导出: {csv_size} 字节")
            
            # JSON
            json_file = os.path.join(tmpdir, "kg.json")
            export_triplets_to_json(triplets, json_file)
            json_size = os.path.getsize(json_file)
            print(f"    ✓ JSON 导出: {json_size} 字节")
            
            # GraphML (可视化)
            graphml_file = os.path.join(tmpdir, "kg.graphml")
            export_graph_to_graphml(graph, graphml_file)
            graphml_size = os.path.getsize(graphml_file)
            print(f"    ✓ GraphML 导出: {graphml_size} 字节")
            
            # RDF Turtle (语义网)
            ttl_file = os.path.join(tmpdir, "kg.ttl")
            export_graph_to_rdf_turtle(triplets, ttl_file)
            ttl_size = os.path.getsize(ttl_file)
            print(f"    ✓ RDF Turtle 导出: {ttl_size} 字节")
            
            # 验证 CSV 格式
            import pandas as pd
            df = pd.read_csv(csv_file, sep="|")
            print(f"    ✓ CSV 格式验证: {len(df)} 行, 列 {list(df.columns)}")
            assert list(df.columns) == ["node_1", "edge", "node_2"], "CSV列名应正确"
        
        print(f"\n  ✅ 完整 Pipeline 导出成功")
        print(f"     原始数据 → 离散化 → 图谱 → 4种格式导出")
        
        return True
    except Exception as e:
        print(f"❌ 导出格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """测试完整集成 - 端到端 Pipeline"""
    print("\n" + "=" * 70)
    print("测试 7: 端到端完整 Pipeline")
    print("=" * 70)
    
    try:
        from DGGS import (
            prepare_for_graph_reasoning
        )
        from examples.polygon_examples import (
            discretize_ssurgo_soil_properties,
            create_ssurgo_sample_data,
        )
        from examples.raster_examples import (
            discretize_cdl_crop_distribution,
            create_cdl_sample_data,
        )
        
        print("  🌍 完整工作流: 原始空间数据 → 知识图谱 → GraphReasoning")
        print()
        
        # === CDL Pipeline ===
        print("  📊 CDL 数据流:")
        print("    1️⃣  生成原始 CDL 栅格像素...")
        cdl_pixels = create_cdl_sample_data(year=2021)
        print(f"       ✓ {len(cdl_pixels)} 个像素 (30m × 30m)")
        
        print("    2️⃣  空间离散化到 DGGS 单元格...")
        cdl_discretized = discretize_cdl_crop_distribution(cdl_pixels, level=12)
        print(f"       ✓ {len(cdl_discretized)} 个 DGGS 单元格")
        
        print("    3️⃣  转换为知识图谱...")
        with tempfile.TemporaryDirectory() as tmpdir:
            cdl_triplets, cdl_graph = prepare_for_graph_reasoning(
                discretized_data=cdl_discretized,
                data_type="cdl",
                output_dir=tmpdir
            )
            print(f"       ✓ {len(cdl_triplets)} 个 RDF 三元组")
            print(f"       ✓ {cdl_graph.number_of_nodes()} 个节点, {cdl_graph.number_of_edges()} 条边")
            
            # 验证导出文件
            files = os.listdir(tmpdir)
            print(f"    4️⃣  导出文件: {len(files)} 个")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(tmpdir, f))
                print(f"       ✓ {f}: {size} 字节")
        
        # === SSURGO Pipeline ===
        print("\n  🌱 SSURGO 数据流:")
        print("    1️⃣  生成原始 SSURGO 地图单元...")
        ssurgo_map_units = create_ssurgo_sample_data()
        print(f"       ✓ {len(ssurgo_map_units)} 个地图单元 (多边形)")
        
        print("    2️⃣  空间离散化...")
        ssurgo_discretized = discretize_ssurgo_soil_properties(
            ssurgo_map_units,
            properties=['pH', 'sand_percent', 'clay_percent'],
            level=12
        )
        print(f"       ✓ {len(ssurgo_discretized)} 个 DGGS 单元格")
        
        print("    3️⃣  转换为知识图谱...")
        with tempfile.TemporaryDirectory() as tmpdir:
            ssurgo_triplets, ssurgo_graph = prepare_for_graph_reasoning(
                discretized_data=ssurgo_discretized,
                data_type="ssurgo",
                output_dir=tmpdir
            )
            print(f"       ✓ {len(ssurgo_triplets)} 个 RDF 三元组")
            print(f"       ✓ {ssurgo_graph.number_of_nodes()} 个节点, {ssurgo_graph.number_of_edges()} 条边")
        
        # === 集成验证 ===
        print("\n  🔗 多源数据集成:")
        from DGGS import merge_into_existing_graph
        integrated_graph = merge_into_existing_graph(
            cdl_graph,
            ssurgo_triplets,
            merge_strategy="union"
        )
        print(f"    ✓ 合并后知识图谱:")
        print(f"      - 节点: {integrated_graph.number_of_nodes()}")
        print(f"      - 边: {integrated_graph.number_of_edges()}")
        print(f"      - 来源: CDL + SSURGO")
        
        assert len(cdl_triplets) > 0, "应生成 CDL 三元组"
        assert len(ssurgo_triplets) > 0, "应生成 SSURGO 三元组"
        assert cdl_graph.number_of_nodes() > 0, "CDL 图应有节点"
        assert integrated_graph.number_of_nodes() >= cdl_graph.number_of_nodes(), "合并后节点数应增加"
        
        print(f"\n  ✅ 端到端 Pipeline 测试通过")
        print(f"     原始栅格/矢量 → DGGS离散化 → RDF三元组 → NetworkX图 → 导出格式")
        
        return True
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("█" * 70)
    print("█ 完整 Pipeline 测试: 空间数据 → 知识图谱")
    print("█ Complete Pipeline: Spatial Data → Knowledge Graph")
    print("█" * 70)
    print("\n流程概览:")
    print("  原始空间数据 (CDL 栅格像素, SSURGO 矢量多边形)")
    print("       ↓")
    print("  DGGS 空间离散化 (S2 层次网格)")
    print("       ↓")
    print("  RDF 三元组生成 (主-谓-宾)")
    print("       ↓")
    print("  知识图谱构建 (NetworkX 有向图)")
    print("       ↓")
    print("  多格式导出 (CSV, GraphML, JSON, RDF Turtle)")
    print("       ↓")
    print("  GraphReasoning 框架集成")
    print()
    
    tests = [
        ("导入验证", test_imports),
        ("SpatialEntity 模型", test_spatial_entity),
        ("SpatialRelationship 模型", test_spatial_relationship),
        ("完整离散化→三元组", test_triplet_generation),
        ("完整图谱创建流程", test_graph_creation),
        ("完整 Pipeline 导出", test_export_formats),
        ("端到端完整集成", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 所有测试通过! 完整 Pipeline 运行正常。")
        print("\n📚 Pipeline 说明:")
        print("  1. 原始数据: CDL 栅格(30m分辨率), SSURGO 矢量多边形")
        print("  2. 离散化: S2 DGGS 层次网格聚合 (level 10-14)")
        print("  3. 三元组: RDF 标准格式 (subject-predicate-object)")
        print("  4. 图谱: NetworkX 有向图, 支持图算法和推理")
        print("  5. 导出: CSV/GraphML/JSON/RDF 多格式支持")
        print("  6. 集成: GraphReasoning 框架无缝对接")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
