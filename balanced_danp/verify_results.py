#!/usr/bin/env python3
"""
简单的结果验证脚本
直接分析运行输出的权重平衡效果
"""

def analyze_balance_results():
    """基于运行时输出的数据分析权重平衡效果"""
    
    print("=" * 60)
    print("平衡权重DANP算法 - 结果分析")
    print("=" * 60)
    
    # 维度配置
    dimensions = ['d1', 'd2', 'd3', 'd4']
    criteria_counts = [2, 5, 3, 1]
    total_criteria = sum(criteria_counts)
    
    # 维度权重计算
    print("\n1. 维度权重分配:")
    print("-" * 40)
    for i, (dim, count) in enumerate(zip(dimensions, criteria_counts)):
        weight = count / total_criteria
        print(f"{dim}: {count}个因素 → {weight:.4f} ({weight*100:.1f}%)")
    
    # 基于运行输出的权重数据
    original_weights = [
        0.128903, 0.145031,  # d1: c11, c12
        0.078923, 0.065705, 0.054372, 0.062178, 0.054725,  # d2: c21-c25
        0.112000, 0.123207, 0.122591,  # d3: c31-c33
        0.052367  # d4: c41
    ]
    
    balanced_weights = [
        0.085557, 0.096261,  # d1: c11, c12
        0.113560, 0.094542, 0.078235, 0.089467, 0.078742,  # d2: c21-c25
        0.085371, 0.093913, 0.093444,  # d3: c31-c33
        0.090909  # d4: c41
    ]
    
    criteria_names = [
        'c11', 'c12',
        'c21', 'c22', 'c23', 'c24', 'c25',
        'c31', 'c32', 'c33',
        'c41'
    ]
    
    print("\n2. 权重对比分析:")
    print("-" * 60)
    print("因素\t\t原始权重\t平衡权重\t变化(%)")
    print("-" * 60)
    
    start_idx = 0
    for dim_idx, (dim, count) in enumerate(zip(dimensions, criteria_counts)):
        print(f"\n{dim} (目标权重: {criteria_counts[dim_idx]/total_criteria:.1%}):")
        
        dim_orig_sum = 0
        dim_bal_sum = 0
        
        for i in range(start_idx, start_idx + count):
            orig = original_weights[i]
            bal = balanced_weights[i]
            change = ((bal - orig) / orig) * 100
            
            dim_orig_sum += orig
            dim_bal_sum += bal
            
            print(f"  {criteria_names[i]}\t\t{orig:.6f}\t{bal:.6f}\t{change:+.1f}%")
        
        print(f"  小计:\t\t{dim_orig_sum:.6f}\t{dim_bal_sum:.6f}")
        start_idx += count
    
    print("\n3. 权重平衡效果评估:")
    print("-" * 40)
    
    # 计算各维度的权重分布
    print("维度权重分布对比:")
    start_idx = 0
    for dim_idx, (dim, count) in enumerate(zip(dimensions, criteria_counts)):
        target_weight = count / total_criteria
        
        orig_sum = sum(original_weights[start_idx:start_idx + count])
        bal_sum = sum(balanced_weights[start_idx:start_idx + count])
        
        print(f"{dim}: 目标{target_weight:.1%} | 原始{orig_sum:.1%} | 平衡{bal_sum:.1%}")
        start_idx += count
    
    print(f"\n权重总和验证:")
    print(f"原始权重总和: {sum(original_weights):.6f}")
    print(f"平衡权重总和: {sum(balanced_weights):.6f}")
    
    print("\n4. 主要改进效果:")
    print("-" * 40)
    print("✓ d1 (2个因素): 权重从27.4%降到18.2% - 消除小维度过度放大")
    print("✓ d2 (5个因素): 权重从31.6%升到45.5% - 避免大维度被稀释")
    print("✓ d3 (3个因素): 权重从35.8%降到27.3% - 调整到合理比例")
    print("✓ d4 (1个因素): 权重从5.2%升到9.1% - 单因素获得基础权重")
    
    print("\n5. 算法特点:")
    print("-" * 40)
    print("• 维度内相对权重关系保持不变")
    print("• 仅调整跨维度的权重分配比例")
    print("• 权重分配更加公平合理")
    print("• 消除了结构性偏差问题")

if __name__ == "__main__":
    analyze_balance_results() 