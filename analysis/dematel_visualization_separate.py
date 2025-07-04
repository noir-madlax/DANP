#!/usr/bin/env python3
"""
DEMATEL因果图可视化脚本 - 分离版本
分别生成散点图和网络图两个独立文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_scatter_plot(result_dir, dr_data, factors, d_plus_r, d_minus_r):
    """
    创建DEMATEL因果分析散点图
    """
    print("正在创建因果分析散点图...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # 创建散点图
    scatter = ax.scatter(d_plus_r, d_minus_r, 
                        s=500, alpha=0.8, 
                        c=range(len(factors)), 
                        cmap='viridis',
                        edgecolors='black',
                        linewidth=2)
    
    # 添加因素标签
    for i, factor in enumerate(factors):
        ax.annotate(factor, (d_plus_r[i], d_minus_r[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, ha='left', va='bottom',
                   fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.4", 
                            facecolor='white', alpha=0.9,
                            edgecolor='gray'))
    
    # 添加参考线
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=3)
    ax.axvline(x=np.mean(d_plus_r), color='red', linestyle='--', alpha=0.8, linewidth=3)
    
    # 设置标签和标题
    ax.set_xlabel('D+R (中心度/重要性)', fontsize=16, fontweight='bold')
    ax.set_ylabel('D-R (原因度)', fontsize=16, fontweight='bold')
    ax.set_title('DEMATEL因果分析散点图', fontsize=18, fontweight='bold', pad=25)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    
    # 添加象限标签
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 调整象限标签位置
    ax.text(xlim[1]*0.82, ylim[1]*0.82, 
             '高重要性\n原因因素\n(关键驱动)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcoral", alpha=0.9,
                      edgecolor='darkred', linewidth=2),
             fontsize=11, fontweight='bold')
    
    ax.text(xlim[0] + (xlim[1]-xlim[0])*0.18, ylim[1]*0.82, 
             '低重要性\n原因因素\n(次要驱动)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgreen", alpha=0.9,
                      edgecolor='darkgreen', linewidth=2),
             fontsize=11, fontweight='bold')
    
    ax.text(xlim[1]*0.82, ylim[0] + (ylim[1]-ylim[0])*0.18, 
             '高重要性\n结果因素\n(关键目标)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9,
                      edgecolor='orange', linewidth=2),
             fontsize=11, fontweight='bold')
    
    ax.text(xlim[0] + (xlim[1]-xlim[0])*0.18, ylim[0] + (ylim[1]-ylim[0])*0.18, 
             '低重要性\n结果因素\n(次要目标)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightblue", alpha=0.9,
                      edgecolor='darkblue', linewidth=2),
             fontsize=11, fontweight='bold')
    
    # 添加数值标注
    ax.text(xlim[1]*0.95, ylim[1]*0.95, 
            f'D+R范围: {d_plus_r.min():.2f}~{d_plus_r.max():.2f}\nD-R范围: {d_minus_r.min():.2f}~{d_minus_r.max():.2f}',
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 调整坐标轴
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 保存散点图
    scatter_file = os.path.join(result_dir, 'dematel_scatter_plot.png')
    plt.tight_layout()
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"散点图已保存到: {scatter_file}")
    
    plt.show()
    plt.close()
    
    return scatter_file

def create_network_plot(result_dir, tc_data, factors, d_plus_r, d_minus_r, threshold=0.1):
    """
    创建DEMATEL网络因果图
    """
    print("正在创建网络因果图...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for factor in factors:
        G.add_node(factor)
    
    # 添加边（基于影响强度）
    edge_count = 0
    for i in range(len(factors)):
        for j in range(len(factors)):
            if i != j:
                influence = tc_data.iloc[i, j]
                if influence > threshold:
                    G.add_edge(factors[i], factors[j], weight=influence)
                    edge_count += 1
    
    print(f"使用阈值 {threshold}，共添加 {edge_count} 条边")
    
    # 如果边数太少，降低阈值
    if edge_count < len(factors):
        new_threshold = np.percentile(tc_data.values[tc_data.values > 0], 65)
        print(f"边数较少，降低阈值到 {new_threshold:.4f}")
        G.clear_edges()
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j:
                    influence = tc_data.iloc[i, j]
                    if influence > new_threshold:
                        G.add_edge(factors[i], factors[j], weight=influence)
        threshold = new_threshold
    
    # 设置节点位置（使用改进的布局）
    pos = nx.spring_layout(G, k=3.5, iterations=100, seed=42)
    
    # 根据D-R值设置节点颜色
    node_colors = []
    for factor in factors:
        idx = factors.index(factor)
        if d_minus_r[idx] > 0:
            node_colors.append('lightcoral')  # 原因因素用红色
        else:
            node_colors.append('lightblue')   # 结果因素用蓝色
    
    # 根据D+R值设置节点大小
    min_size, max_size = 1200, 4000
    norm_importance = (d_plus_r - d_plus_r.min()) / (d_plus_r.max() - d_plus_r.min())
    node_sizes = [min_size + (max_size - min_size) * norm for norm in norm_importance]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=3)
    
    # 绘制边
    if G.edges():
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights)
        min_weight = min(weights)
        
        # 标准化权重用于边宽度和透明度
        edge_widths = [1 + (w/max_weight) * 5 for w in weights]
        edge_alphas = [0.3 + (w - min_weight)/(max_weight - min_weight) * 0.5 for w in weights]
        
        # 绘制边
        for i, (u, v) in enumerate(edges):
            nx.draw_networkx_edges(G, pos, ax=ax, 
                                 edgelist=[(u, v)],
                                 edge_color='gray',
                                 width=edge_widths[i],
                                 alpha=edge_alphas[i],
                                 arrows=True,
                                 arrowsize=25,
                                 arrowstyle='->',
                                 connectionstyle="arc3,rad=0.1")
    
    # 添加节点标签
    labels = {}
    for factor in factors:
        idx = factors.index(factor)
        labels[factor] = f"{factor}\n({d_plus_r[idx]:.2f})"
    
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')
    
    # 设置标题
    ax.set_title('DEMATEL网络因果图', fontsize=18, fontweight='bold', pad=25)
    ax.axis('off')
    
    # 添加详细图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=15, label='原因因素 (D-R > 0)', markeredgecolor='black',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=15, label='结果因素 (D-R ≤ 0)', markeredgecolor='black',
                   markeredgewidth=2),
        plt.Line2D([0], [0], color='gray', linewidth=4, alpha=0.7,
                   label=f'影响关系 (阈值 > {threshold:.3f})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markersize=12, label='节点大小 ∝ 重要性(D+R)', markeredgecolor='black')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
             frameon=True, fancybox=True, shadow=True)
    
    # 添加统计信息
    stats_text = f"""网络统计:
节点数: {len(factors)}
边数: {len(G.edges())}
原因因素: {len([f for f in factors if d_minus_r[factors.index(f)] > 0])}个
结果因素: {len([f for f in factors if d_minus_r[factors.index(f)] <= 0])}个"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                                            facecolor="white", alpha=0.8))
    
    # 保存网络图
    network_file = os.path.join(result_dir, 'dematel_network_plot.png')
    plt.tight_layout()
    plt.savefig(network_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"网络图已保存到: {network_file}")
    
    plt.show()
    plt.close()
    
    return network_file

def create_separate_diagrams(result_dir, threshold=0.1):
    """
    创建分离的DEMATEL因果图
    """
    
    print(f"正在处理目录: {result_dir}")
    
    # 文件路径
    dr_file = os.path.join(result_dir, 'D_R.xlsx')
    tc_file = os.path.join(result_dir, 'Tc_defuzzied.xlsx')
    
    # 检查文件是否存在
    if not os.path.exists(dr_file):
        print(f"错误: 未找到文件 {dr_file}")
        return
    if not os.path.exists(tc_file):
        print(f"错误: 未找到文件 {tc_file}")
        return
    
    try:
        # 读取数据
        print("正在读取D_R.xlsx文件...")
        dr_data = pd.read_excel(dr_file, index_col=0)
        print(f"D_R数据形状: {dr_data.shape}")
        print(f"D_R数据列名: {list(dr_data.columns)}")
        
        print("正在读取Tc_defuzzied.xlsx文件...")
        tc_data = pd.read_excel(tc_file, index_col=0)
        print(f"Tc数据形状: {tc_data.shape}")
        
        # 获取因素名称
        factors = dr_data.index.tolist()
        print(f"因素列表: {factors}")
        
        # 提取D+R和D-R数据
        d_plus_r = dr_data['D+R'].values
        d_minus_r = dr_data['D-R'].values
        
        print(f"D+R范围: {d_plus_r.min():.4f} ~ {d_plus_r.max():.4f}")
        print(f"D-R范围: {d_minus_r.min():.4f} ~ {d_minus_r.max():.4f}")
        
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return
    
    # 分别创建两个图
    print("\n" + "="*60)
    scatter_file = create_scatter_plot(result_dir, dr_data, factors, d_plus_r, d_minus_r)
    
    print("\n" + "="*60)  
    network_file = create_network_plot(result_dir, tc_data, factors, d_plus_r, d_minus_r, threshold)
    
    return scatter_file, network_file

def analyze_causal_relationships(result_dir):
    """分析因果关系并生成文字报告"""
    
    dr_file = os.path.join(result_dir, 'D_R.xlsx')
    
    try:
        dr_data = pd.read_excel(dr_file, index_col=0)
        factors = dr_data.index.tolist()
        
        print(f"\n{'='*60}")
        print("DEMATEL因果关系分析报告")
        print(f"{'='*60}")
        
        # 按重要性排序
        importance_ranking = dr_data.sort_values('D+R', ascending=False)
        print(f"\n1. 重要性排序 (D+R值):")
        print("-" * 40)
        for i, (factor, row) in enumerate(importance_ranking.iterrows(), 1):
            print(f"{i:2d}. {factor:<10} D+R = {row['D+R']:.4f}")
        
        # 原因因素分析
        cause_factors = dr_data[dr_data['D-R'] > 0].sort_values('D-R', ascending=False)
        print(f"\n2. 原因因素 (D-R > 0，共{len(cause_factors)}个):")
        print("-" * 40)
        if len(cause_factors) > 0:
            for factor, row in cause_factors.iterrows():
                status = "强" if row['D-R'] > 0.5 else "中" if row['D-R'] > 0.1 else "弱"
                print(f"   {factor:<10} D-R = {row['D-R']:+.4f} [{status}驱动力], 重要性 = {row['D+R']:.4f}")
        else:
            print("   无原因因素")
        
        # 结果因素分析  
        result_factors = dr_data[dr_data['D-R'] <= 0].sort_values('D-R', ascending=True)
        print(f"\n3. 结果因素 (D-R ≤ 0，共{len(result_factors)}个):")
        print("-" * 40)
        if len(result_factors) > 0:
            for factor, row in result_factors.iterrows():
                status = "强" if row['D-R'] < -0.5 else "中" if row['D-R'] < -0.1 else "弱"
                print(f"   {factor:<10} D-R = {row['D-R']:+.4f} [{status}被影响], 重要性 = {row['D+R']:.4f}")
        else:
            print("   无结果因素")
        
        # 关键因素识别
        print(f"\n4. 关键因素识别:")
        print("-" * 40)
        
        # 最重要的原因因素
        if len(cause_factors) > 0:
            key_cause = cause_factors.iloc[0]
            print(f"   🔑 最关键原因因素: {key_cause.name}")
            print(f"      → 最强驱动力 (D-R = {key_cause['D-R']:+.4f})")
            print(f"      → 重要性排名: #{importance_ranking.index.get_loc(key_cause.name) + 1}")
        
        # 最重要的结果因素
        if len(result_factors) > 0:
            key_result = result_factors.iloc[0]
            print(f"   🎯 最关键结果因素: {key_result.name}")
            print(f"      → 最强被影响性 (D-R = {key_result['D-R']:+.4f})")
            print(f"      → 重要性排名: #{importance_ranking.index.get_loc(key_result.name) + 1}")
        
        # 整体最重要因素
        most_important = importance_ranking.iloc[0]
        factor_type = "原因" if most_important['D-R'] > 0 else "结果"
        print(f"   ⭐ 整体最重要因素: {most_important.name} [{factor_type}因素]")
        print(f"      → 系统中最活跃 (D+R = {most_important['D+R']:.4f})")
        
    except Exception as e:
        print(f"分析因果关系时出错: {e}")

def main():
    """主函数"""
    
    # 指定目录
    result_dir = 'result/20250704_121352'
    
    # 检查目录是否存在
    if not os.path.exists(result_dir):
        print(f"错误: 目录 {result_dir} 不存在")
        return
    
    print(f"开始处理DEMATEL结果目录: {result_dir}")
    print("="*60)
    
    # 创建分离的因果图
    files = create_separate_diagrams(result_dir, threshold=0.1)
    
    if files:
        scatter_file, network_file = files
        print(f"\n✅ 图像文件生成成功:")
        print(f"   📊 散点图: {os.path.basename(scatter_file)}")
        print(f"   🕸️  网络图: {os.path.basename(network_file)}")
        
        # 分析因果关系
        analyze_causal_relationships(result_dir)
        
        print(f"\n✅ 所有文件已生成到目录: {result_dir}")
    else:
        print("因果图创建失败!")

if __name__ == "__main__":
    main() 