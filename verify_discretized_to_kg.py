#!/usr/bin/env python
"""
简单验证离散化数据到知识图谱模块
Quick verification that the discretized_to_kg module works correctly
"""

import sys

def main():
    print("\n🔍 验证离散化数据到知识图谱模块")
    print("=" * 70)
    
    # Test 1: Imports
    print("\n✅ 测试 1/3: 导入模块...")
    try:
        from Dggs import (
            SpatialEntity,
            create_knowledge_graph_from_discretized_data,
            export_triplets_to_csv,
            prepare_for_graph_reasoning
        )
        from examples.raster_examples import discretized_cdl_to_triplets
        from examples.polygon_examples import discretized_ssurgo_to_triplets
        
        print("   ✓ 所有导入成功")
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    
    # Test 2: CDL workflow
    print("\n✅ 测试 2/3: CDL 工作流...")
    try:
        from examples.raster_examples import (
            discretize_cdl_crop_distribution,
            create_cdl_sample_data,
        )
        
        pixels = create_cdl_sample_data(year=2021)
        cdl_result = discretize_cdl_crop_distribution(pixels, level=12)
        
        first_cell = list(cdl_result.keys())[0]
        first_data = cdl_result[first_cell]
        
        triplets = discretized_cdl_to_triplets(first_cell, first_data)
        
        print(f"   ✓ 离散化了 {len(cdl_result)} 个 CDL 单元格")
        print(f"   ✓ 生成了 {len(triplets)} 个三元组")
        
        # Verify triplet structure
        assert len(triplets) > 0, "应生成三元组"
        assert all(len(t) == 3 for t in triplets), "三元组应有3个元素"
        print(f"   ✓ 三元组格式正确")
        
    except Exception as e:
        print(f"   ✗ CDL 工作流失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Graph creation and export
    print("\n✅ 测试 3/3: 知识图谱创建和导出...")
    try:
        graph, all_triplets = create_knowledge_graph_from_discretized_data(
            cdl_result,
            data_type="cdl"
        )
        
        print(f"   ✓ 图谱节点数: {graph.number_of_nodes()}")
        print(f"   ✓ 图谱边数: {graph.number_of_edges()}")
        print(f"   ✓ 三元组总数: {len(all_triplets)}")
        
        assert graph.number_of_nodes() > 0, "图应有节点"
        assert graph.number_of_edges() > 0, "图应有边"
        
        # Test export
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test.csv")
            export_triplets_to_csv(all_triplets, output_file)
            
            assert os.path.exists(output_file), "CSV 文件应生成"
            file_size = os.path.getsize(output_file)
            print(f"   ✓ CSV 导出成功 ({file_size} 字节)")
        
    except Exception as e:
        print(f"   ✗ 图创建/导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！模块运行正常。")
    print("=" * 70)
    
    print("\n💡 下一步:")
    print("   1. 运行完整示例: python examples/discretized_to_kg_examples.py")
    print("   2. 查看文档: cat DISCRETIZED_TO_KG_GUIDE.md")
    print("   3. 查看快速参考: cat DISCRETIZED_TO_KG_QUICK_REFERENCE.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
