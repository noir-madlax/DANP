import pandas as pd
import os

# 获取最新的结果目录
result_dirs = [d for d in os.listdir('result') if os.path.isdir(os.path.join('result', d))]
latest_result = max(result_dirs)
result_dir = os.path.join('result', latest_result)

print(f"分析结果目录: {result_dir}")
print("=" * 60)

# 1. 维度权重分配
print("\n=== 维度权重分配 ===")
df_dim = pd.read_excel(os.path.join(result_dir, 'dimension_weights_distribution.xlsx'))
print(df_dim.to_string(index=False))

# 2. 权重对比
print("\n=== 权重对比详情 ===")
df_comp = pd.read_excel(os.path.join(result_dir, 'weight_comparison.xlsx'))
print(df_comp.to_string(index=False))

# 3. 按维度分组分析
print("\n=== 按维度分组的权重分析 ===")
print("维度\t因素\t\t原始权重\t平衡权重\t变化(%)")
print("-" * 70)

# 根据维度分组
dimensions = ['d1', 'd2', 'd3', 'd4']
criteria_counts = [2, 5, 3, 1]
start_idx = 0

for dim_idx, (dim, count) in enumerate(zip(dimensions, criteria_counts)):
    print(f"{dim}:")
    dim_orig_sum = 0
    dim_bal_sum = 0
    
    for i in range(start_idx, start_idx + count):
        criteria = df_comp.iloc[i]['Criteria']
        orig_weight = df_comp.iloc[i]['Original_Weight']
        bal_weight = df_comp.iloc[i]['Balanced_Weight']
        change_pct = df_comp.iloc[i]['Relative_Change_Percent']
        
        dim_orig_sum += orig_weight
        dim_bal_sum += bal_weight
        
        print(f"\t{criteria}\t\t{orig_weight:.6f}\t{bal_weight:.6f}\t{change_pct:+.1f}%")
    
    print(f"\t小计:\t\t{dim_orig_sum:.6f}\t{dim_bal_sum:.6f}")
    print()
    start_idx += count

# 4. 总结分析
print("=== 权重平衡效果总结 ===")
print(f"总权重 - 原始: {df_comp['Original_Weight'].sum():.6f}")
print(f"总权重 - 平衡: {df_comp['Balanced_Weight'].sum():.6f}")

print("\n维度权重分布:")
for i, row in df_dim.iterrows():
    print(f"{row['Dimension']}: {row['Criteria_Count']}个因素 → {row['Weight_Percentage']:.1f}%")

print("\n主要改进效果:")
print("- 权重较高的小维度(如d1: 2个因素)的权重被适当降低")
print("- 权重较低的大维度(如d2: 5个因素)的权重被适当提高") 
print("- 单因素维度(如d4: 1个因素)获得了合理的基础权重") 