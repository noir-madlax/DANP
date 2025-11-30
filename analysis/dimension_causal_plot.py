#!/usr/bin/env python3
"""
DEMATEL维度级因果图可视化脚本
专门绘制维度级的因果关系散点图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_dimension_scatter_plot(result_dir):
    """
    创建维度级DEMATEL因果分析散点图
    """
    print("正在创建维度级因果分析散点图...")
    
    # 文件路径
    td_dr_file = os.path.join(result_dir, 'Td_D_R.xlsx')
    
    # 检查文件是否存在
    if not os.path.exists(td_dr_file):
        print(f"错误: 未找到文件 {td_dr_file}")
        return None
    
    try:
        # 读取维度级数据
        print("正在读取Td_D_R.xlsx文件...")
        td_dr_data = pd.read_excel(td_dr_file, index_col=0)
        print(f"维度D_R数据形状: {td_dr_data.shape}")
        print(f"维度D_R数据列名: {list(td_dr_data.columns)}")
        
        # 获取维度名称并转换为大写C
        dimensions = [dim.replace('d', 'C') for dim in td_dr_data.index.tolist()]
        print(f"维度列表: {dimensions}")
        
        # 提取D+R和D-R数据
        d_plus_r = td_dr_data['D+R'].values
        d_minus_r = td_dr_data['D-R'].values
        
        print(f"维度D+R范围: {d_plus_r.min():.4f} ~ {d_plus_r.max():.4f}")
        print(f"维度D-R范围: {d_minus_r.min():.4f} ~ {d_minus_r.max():.4f}")
        
    except Exception as e:
        print(f"读取维度数据时出错: {e}")
        return None
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 为不同维度设置不同颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    dimension_colors = [colors[i % len(colors)] for i in range(len(dimensions))]
    
    # 创建散点图
    scatter = ax.scatter(d_plus_r, d_minus_r, 
                        s=800, alpha=0.8, 
                        c=dimension_colors,
                        edgecolors='black',
                        linewidth=3)
    
    # 添加维度标签
    for i, dimension in enumerate(dimensions):
        ax.annotate(dimension, (d_plus_r[i], d_minus_r[i]), 
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=20, ha='left', va='bottom',
                   fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor='white', alpha=0.9,
                            edgecolor=dimension_colors[i], linewidth=2))
    
    # 添加参考线
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=3)
    ax.axvline(x=np.mean(d_plus_r), color='red', linestyle='--', alpha=0.8, linewidth=3)
    
    # 设置标签
    ax.set_xlabel('中心度（D + R）', fontsize=20, fontweight='bold')
    ax.set_ylabel('原因度（D - R）', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=1)
    
    # 添加象限标签 - 放置在每个象限的中心
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 计算参考线位置
    x_mid = np.mean(d_plus_r)
    y_mid = 0
    
    # 右上象限 - 高重要性原因维度
    x_right_center = (x_mid + xlim[1]) / 2
    y_top_center = (y_mid + ylim[1]) / 2
    ax.text(x_right_center, y_top_center, 
             '高重要性\n原因维度\n(核心驱动)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcoral", alpha=0.9,
                      edgecolor='darkred', linewidth=3),
             fontsize=20, fontweight='bold')
    
    # 左上象限 - 低重要性原因维度
    x_left_center = (xlim[0] + x_mid) / 2
    ax.text(x_left_center, y_top_center, 
             '低重要性\n原因维度\n(辅助驱动)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.9,
                      edgecolor='darkgreen', linewidth=3),
             fontsize=20, fontweight='bold')
    
    # 右下象限 - 高重要性结果维度
    y_bottom_center = (ylim[0] + y_mid) / 2
    ax.text(x_right_center, y_bottom_center, 
             '高重要性\n结果维度\n(关键输出)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9,
                      edgecolor='orange', linewidth=3),
             fontsize=20, fontweight='bold')
    
    # 左下象限 - 低重要性结果维度
    ax.text(x_left_center, y_bottom_center, 
             '低重要性\n结果维度\n(次要输出)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9,
                      edgecolor='darkblue', linewidth=3),
             fontsize=20, fontweight='bold')
    


    
    # 调整坐标轴
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # 设置坐标轴范围
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(0.4, 1.4)
    
    # 创建输出目录
    output_dir = os.path.join(result_dir, 'dimension_analysis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 保存散点图
    scatter_file = os.path.join(output_dir, 'dimension_scatter_plot.png')
    plt.tight_layout()
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"维度散点图已保存到: {scatter_file}")
    
    plt.close()
    
    return scatter_file, td_dr_data

def analyze_dimension_relationships(result_dir, td_dr_data):
    """分析维度级因果关系并生成报告"""
    
    try:
        dimensions = td_dr_data.index.tolist()
        
        print(f"\n{'='*60}")
        print("DEMATEL维度级因果关系分析报告")
        print(f"{'='*60}")
        
        # 按重要性排序
        importance_ranking = td_dr_data.sort_values('D+R', ascending=False)
        print(f"\n1. 维度重要性排序 (D+R值):")
        print("-" * 40)
        for i, (dimension, row) in enumerate(importance_ranking.iterrows(), 1):
            print(f"{i:2d}. {dimension:<6} D+R = {row['D+R']:.4f}")
        
        # 原因维度分析
        cause_dimensions = td_dr_data[td_dr_data['D-R'] > 0].sort_values('D-R', ascending=False)
        print(f"\n2. 原因维度 (D-R > 0，共{len(cause_dimensions)}个):")
        print("-" * 40)
        if len(cause_dimensions) > 0:
            for dimension, row in cause_dimensions.iterrows():
                status = "强" if row['D-R'] > 0.5 else "中" if row['D-R'] > 0.2 else "弱"
                print(f"   {dimension:<6} D-R = {row['D-R']:+.4f} [{status}驱动力], 重要性 = {row['D+R']:.4f}")
        else:
            print("   无原因维度")
        
        # 结果维度分析  
        result_dimensions = td_dr_data[td_dr_data['D-R'] <= 0].sort_values('D-R', ascending=True)
        print(f"\n3. 结果维度 (D-R ≤ 0，共{len(result_dimensions)}个):")
        print("-" * 40)
        if len(result_dimensions) > 0:
            for dimension, row in result_dimensions.iterrows():
                status = "强" if row['D-R'] < -0.5 else "中" if row['D-R'] < -0.2 else "弱"
                print(f"   {dimension:<6} D-R = {row['D-R']:+.4f} [{status}被影响], 重要性 = {row['D+R']:.4f}")
        else:
            print("   无结果维度")
        
        # 维度级关键发现
        print(f"\n4. 维度级关键发现:")
        print("-" * 40)
        
        # 最重要的原因维度
        if len(cause_dimensions) > 0:
            key_cause = cause_dimensions.iloc[0]
            print(f"   🔑 最关键原因维度: {key_cause.name}")
            print(f"      → 最强驱动力 (D-R = {key_cause['D-R']:+.4f})")
            print(f"      → 重要性排名: #{importance_ranking.index.get_loc(key_cause.name) + 1}")
        
        # 最重要的结果维度
        if len(result_dimensions) > 0:
            key_result = result_dimensions.iloc[0]
            print(f"   🎯 最关键结果维度: {key_result.name}")
            print(f"      → 最强被影响性 (D-R = {key_result['D-R']:+.4f})")
            print(f"      → 重要性排名: #{importance_ranking.index.get_loc(key_result.name) + 1}")
        
        # 整体最重要维度
        most_important = importance_ranking.iloc[0]
        dimension_type = "原因" if most_important['D-R'] > 0 else "结果"
        print(f"   ⭐ 整体最重要维度: {most_important.name} [{dimension_type}维度]")
        print(f"      → 系统中最活跃 (D+R = {most_important['D+R']:.4f})")
        
        # 维度级策略建议
        print(f"\n5. 维度级策略建议:")
        print("-" * 40)
        
        if len(cause_dimensions) > 0:
            print(f"   📈 重点投资维度: {', '.join(cause_dimensions.index[:2].tolist())}")
            print(f"      → 这些维度能够驱动整个系统的改进")
        
        if len(result_dimensions) > 0:
            key_outputs = result_dimensions.nlargest(2, 'D+R')
            print(f"   🎯 重点监控维度: {', '.join(key_outputs.index.tolist())}")
            print(f"      → 这些维度是系统的关键输出指标")
        
        # 保存维度分析报告
        output_dir = os.path.join(result_dir, 'dimension_analysis')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        report_file = os.path.join(output_dir, 'dimension_causal_analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DEMATEL维度级因果关系分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 维度重要性排序 (D+R值):\n")
            f.write("-" * 40 + "\n")
            for i, (dimension, row) in enumerate(importance_ranking.iterrows(), 1):
                f.write(f"{i:2d}. {dimension:<6} D+R = {row['D+R']:.4f}\n")
            
            f.write(f"\n2. 原因维度 (D-R > 0，共{len(cause_dimensions)}个):\n")
            f.write("-" * 40 + "\n")
            for dimension, row in cause_dimensions.iterrows():
                f.write(f"   {dimension:<6} D-R = {row['D-R']:+.4f}, 重要性 = {row['D+R']:.4f}\n")
            
            f.write(f"\n3. 结果维度 (D-R ≤ 0，共{len(result_dimensions)}个):\n")
            f.write("-" * 40 + "\n")
            for dimension, row in result_dimensions.iterrows():
                f.write(f"   {dimension:<6} D-R = {row['D-R']:+.4f}, 重要性 = {row['D+R']:.4f}\n")
        
        print(f"\n维度分析报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"分析维度因果关系时出错: {e}")

def main():
    """主函数"""
    
    # 指定目录
    result_dir = 'result/20251026_175633'
    
    # 检查目录是否存在
    if not os.path.exists(result_dir):
        print(f"错误: 目录 {result_dir} 不存在")
        return
    
    print(f"开始处理DEMATEL维度级结果目录: {result_dir}")
    print("="*60)
    
    # 创建维度级因果图
    result = create_dimension_scatter_plot(result_dir)
    
    if result:
        scatter_file, td_dr_data = result
        print(f"\n✅ 维度散点图生成成功:")
        print(f"   📊 维度散点图: {os.path.basename(scatter_file)}")
        
        # 分析维度因果关系
        analyze_dimension_relationships(result_dir, td_dr_data)
        
        print(f"\n✅ 维度级分析完成，文件已生成到目录: {result_dir}")
    else:
        print("维度因果图创建失败!")

if __name__ == "__main__":
    main() 