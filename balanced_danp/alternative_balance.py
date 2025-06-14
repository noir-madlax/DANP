#!/usr/bin/env python3
"""
替代的权重平衡方法
提供多种平衡策略供用户选择
"""

import numpy as np

def calculate_alternative_balance_methods(original_weights, criteria_counts, dimension_names):
    """
    计算多种权重平衡方法
    
    Args:
        original_weights: 原始权重数组
        criteria_counts: 各维度因素数量列表
        dimension_names: 维度名称列表
    
    Returns:
        dict: 包含不同平衡方法结果的字典
    """
    
    total_criteria = sum(criteria_counts)
    dimension_count = len(criteria_counts)
    
    results = {}
    
    # 方法1: 当前实现 - 完全按因素数量比例分配
    print("=== 方法1: 完全比例分配 (当前实现) ===")
    method1_weights = original_weights.copy()
    
    criteria_start_index = 0
    for dim_idx in range(dimension_count):
        dim_criteria_count = criteria_counts[dim_idx]
        dim_end_index = criteria_start_index + dim_criteria_count
        
        # 维度目标权重
        dim_target_weight = criteria_counts[dim_idx] / total_criteria
        
        # 获取该维度内所有因素的权重
        dim_weights = original_weights[criteria_start_index:dim_end_index]
        dim_weight_sum = np.sum(dim_weights)
        
        if dim_weight_sum > 0:
            # 计算维度内各因素的相对权重
            dim_relative_weights = dim_weights / dim_weight_sum
            # 应用维度目标权重
            balanced_dim_weights = dim_relative_weights * dim_target_weight
            method1_weights[criteria_start_index:dim_end_index] = balanced_dim_weights
        
        criteria_start_index = dim_end_index
    
    results['method1'] = method1_weights
    
    # 方法2: 保守平衡 - 仅调整过度偏差
    print("\n=== 方法2: 保守平衡 - 仅调整过度偏差 ===")
    method2_weights = original_weights.copy()
    
    # 计算各维度的原始权重和理论权重
    criteria_start_index = 0
    dimension_adjustments = []
    
    for dim_idx in range(dimension_count):
        dim_criteria_count = criteria_counts[dim_idx]
        dim_end_index = criteria_start_index + dim_criteria_count
        
        # 原始维度权重
        dim_original_weight = np.sum(original_weights[criteria_start_index:dim_end_index])
        # 理论维度权重
        dim_theoretical_weight = criteria_counts[dim_idx] / total_criteria
        
        # 计算偏差程度
        deviation = dim_original_weight - dim_theoretical_weight
        deviation_ratio = abs(deviation) / dim_theoretical_weight
        
        print(f"{dimension_names[dim_idx]}: 原始{dim_original_weight:.1%} vs 理论{dim_theoretical_weight:.1%}, 偏差{deviation:+.1%} ({deviation_ratio:.1%})")
        
        # 只调整偏差超过20%的维度
        if deviation_ratio > 0.2:
            adjustment_factor = 0.7  # 调整70%的偏差
            target_weight = dim_original_weight - deviation * adjustment_factor
        else:
            target_weight = dim_original_weight  # 不调整
        
        dimension_adjustments.append(target_weight)
        criteria_start_index = dim_end_index
    
    # 重新标准化调整后的维度权重
    total_adjusted = sum(dimension_adjustments)
    dimension_adjustments = [w / total_adjusted for w in dimension_adjustments]
    
    # 应用调整
    criteria_start_index = 0
    for dim_idx in range(dimension_count):
        dim_criteria_count = criteria_counts[dim_idx]
        dim_end_index = criteria_start_index + dim_criteria_count
        
        dim_weights = original_weights[criteria_start_index:dim_end_index]
        dim_weight_sum = np.sum(dim_weights)
        
        if dim_weight_sum > 0:
            dim_relative_weights = dim_weights / dim_weight_sum
            balanced_dim_weights = dim_relative_weights * dimension_adjustments[dim_idx]
            method2_weights[criteria_start_index:dim_end_index] = balanced_dim_weights
        
        criteria_start_index = dim_end_index
    
    results['method2'] = method2_weights
    
    # 方法3: 混合平衡 - 考虑原始重要性
    print("\n=== 方法3: 混合平衡 - 原始重要性 + 结构平衡 ===")
    method3_weights = original_weights.copy()
    
    # 50%考虑原始权重，50%考虑结构平衡
    alpha = 0.5  # 平衡系数
    
    criteria_start_index = 0
    for dim_idx in range(dimension_count):
        dim_criteria_count = criteria_counts[dim_idx]
        dim_end_index = criteria_start_index + dim_criteria_count
        
        # 原始维度权重
        dim_original_weight = np.sum(original_weights[criteria_start_index:dim_end_index])
        # 理论维度权重
        dim_theoretical_weight = criteria_counts[dim_idx] / total_criteria
        
        # 混合目标权重
        dim_target_weight = alpha * dim_theoretical_weight + (1 - alpha) * dim_original_weight
        
        dim_weights = original_weights[criteria_start_index:dim_end_index]
        dim_weight_sum = np.sum(dim_weights)
        
        if dim_weight_sum > 0:
            dim_relative_weights = dim_weights / dim_weight_sum
            balanced_dim_weights = dim_relative_weights * dim_target_weight
            method3_weights[criteria_start_index:dim_end_index] = balanced_dim_weights
        
        criteria_start_index = dim_end_index
    
    # 重新标准化
    method3_weights = method3_weights / np.sum(method3_weights)
    results['method3'] = method3_weights
    
    return results

def compare_balance_methods():
    """对比不同的平衡方法"""
    
    # 基于实际运行数据
    original_weights = np.array([
        0.128903, 0.145031,  # d1: c11, c12
        0.078923, 0.065705, 0.054372, 0.062178, 0.054725,  # d2: c21-c25
        0.112000, 0.123207, 0.122591,  # d3: c31-c33
        0.052367  # d4: c41
    ])
    
    criteria_counts = [2, 5, 3, 1]
    dimension_names = ['d1', 'd2', 'd3', 'd4']
    criteria_names = [
        'c11', 'c12',
        'c21', 'c22', 'c23', 'c24', 'c25',
        'c31', 'c32', 'c33',
        'c41'
    ]
    
    print("原始权重分布:")
    start_idx = 0
    for dim_idx, (dim, count) in enumerate(zip(dimension_names, criteria_counts)):
        dim_weight = np.sum(original_weights[start_idx:start_idx + count])
        theoretical = count / sum(criteria_counts)
        print(f"{dim}: {dim_weight:.1%} (理论: {theoretical:.1%})")
        start_idx += count
    
    print("\n" + "="*60)
    
    # 计算不同平衡方法
    results = calculate_alternative_balance_methods(original_weights, criteria_counts, dimension_names)
    
    print("\n" + "="*60)
    print("结果对比:")
    print("="*60)
    
    print(f"{'因素':<8} {'原始权重':<12} {'方法1':<12} {'方法2':<12} {'方法3':<12}")
    print("-" * 60)
    
    for i, criteria in enumerate(criteria_names):
        orig = original_weights[i]
        m1 = results['method1'][i]
        m2 = results['method2'][i]
        m3 = results['method3'][i]
        print(f"{criteria:<8} {orig:<12.6f} {m1:<12.6f} {m2:<12.6f} {m3:<12.6f}")
    
    print("\n维度权重分布对比:")
    print("-" * 60)
    print(f"{'维度':<6} {'理论':<8} {'原始':<8} {'方法1':<8} {'方法2':<8} {'方法3':<8}")
    print("-" * 60)
    
    start_idx = 0
    for dim_idx, (dim, count) in enumerate(zip(dimension_names, criteria_counts)):
        theoretical = count / sum(criteria_counts)
        orig_sum = np.sum(original_weights[start_idx:start_idx + count])
        m1_sum = np.sum(results['method1'][start_idx:start_idx + count])
        m2_sum = np.sum(results['method2'][start_idx:start_idx + count])
        m3_sum = np.sum(results['method3'][start_idx:start_idx + count])
        
        print(f"{dim:<6} {theoretical:<8.1%} {orig_sum:<8.1%} {m1_sum:<8.1%} {m2_sum:<8.1%} {m3_sum:<8.1%}")
        start_idx += count
    
    print("\n方法说明:")
    print("方法1 (完全比例): 严格按因素数量比例分配维度权重")
    print("方法2 (保守调整): 仅调整偏差过大(>20%)的维度") 
    print("方法3 (混合平衡): 50%考虑结构平衡 + 50%保持原始重要性")

if __name__ == "__main__":
    compare_balance_methods() 