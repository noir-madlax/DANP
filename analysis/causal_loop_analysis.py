#!/usr/bin/env python3
"""
因果回路分析脚本
基于DEMATEL方法计算强关联矩阵并识别关键回路
根据公式 r_ij = {1 if t_ij >= λ; 0 if t_ij < λ} 构建强关联矩阵
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from datetime import datetime
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CausalLoopAnalyzer:
    def __init__(self, data_dir):
        """
        初始化因果回路分析器
        
        参数:
        data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.tc_data = None
        self.dr_data = None
        self.factors = []
        self.strong_relation_matrix = None
        self.threshold = None
        
    def load_data(self):
        """加载DEMATEL分析结果数据"""
        try:
            # 加载总关系矩阵
            tc_file = os.path.join(self.data_dir, 'Tc_defuzzied.xlsx')
            self.tc_data = pd.read_excel(tc_file, index_col=0)
            
            # 加载D-R数据
            dr_file = os.path.join(self.data_dir, 'D_R.xlsx')
            self.dr_data = pd.read_excel(dr_file, index_col=0)
            
            # 获取因素名称
            self.factors = self.tc_data.index.tolist()
            
            print(f"成功加载数据:")
            print(f"- 因素数量: {len(self.factors)}")
            print(f"- 总关系矩阵形状: {self.tc_data.shape}")
            print(f"- 因素列表: {self.factors}")
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
        return True
    
    def calculate_threshold(self, method='percentile', percentile=70, manual_threshold=None):
        """
        计算或设置阈值λ
        
        参数:
        method: 'percentile', 'mean', 'median', 'manual'
        percentile: 百分位数(当method='percentile'时)
        manual_threshold: 手动设置的阈值
        """
        if method == 'manual' and manual_threshold is not None:
            self.threshold = manual_threshold
        elif method == 'percentile':
            # 使用百分位数方法，忽略对角线元素
            non_diagonal = self.tc_data.values[~np.eye(len(self.factors), dtype=bool)]
            self.threshold = np.percentile(non_diagonal[non_diagonal > 0], percentile)
        elif method == 'mean':
            non_diagonal = self.tc_data.values[~np.eye(len(self.factors), dtype=bool)]
            self.threshold = np.mean(non_diagonal[non_diagonal > 0])
        elif method == 'median':
            non_diagonal = self.tc_data.values[~np.eye(len(self.factors), dtype=bool)]
            self.threshold = np.median(non_diagonal[non_diagonal > 0])
        
        print(f"设置阈值λ = {self.threshold:.4f} (方法: {method})")
        return self.threshold
    
    def build_strong_relation_matrix(self):
        """
        构建强关联矩阵R
        根据公式: r_ij = {1 if t_ij >= λ; 0 if t_ij < λ}
        """
        if self.threshold is None:
            print("错误: 请先设置阈值!")
            return None
        
        # 创建强关联矩阵
        self.strong_relation_matrix = np.zeros_like(self.tc_data.values)
        
        # 应用阈值规则
        for i in range(len(self.factors)):
            for j in range(len(self.factors)):
                if i != j:  # 排除对角线元素
                    if self.tc_data.iloc[i, j] >= self.threshold:
                        self.strong_relation_matrix[i, j] = 1
                    else:
                        self.strong_relation_matrix[i, j] = 0
        
        # 转换为DataFrame
        self.strong_relation_matrix = pd.DataFrame(
            self.strong_relation_matrix,
            index=self.factors,
            columns=self.factors
        )
        
        # 统计强关联关系数量
        strong_relations_count = np.sum(self.strong_relation_matrix.values)
        total_possible = len(self.factors) * (len(self.factors) - 1)
        
        print(f"强关联矩阵构建完成:")
        print(f"- 强关联关系数量: {strong_relations_count}")
        print(f"- 总可能关系数量: {total_possible}")
        print(f"- 强关联比例: {strong_relations_count/total_possible:.2%}")
        
        return self.strong_relation_matrix
    
    def identify_mutual_strong_relations(self):
        """识别互为强因果关系的因素对"""
        if self.strong_relation_matrix is None:
            print("错误: 请先构建强关联矩阵!")
            return []
        
        mutual_relations = []
        
        for i in range(len(self.factors)):
            for j in range(i+1, len(self.factors)):  # 避免重复检查
                # 检查是否互为强因果关系
                if (self.strong_relation_matrix.iloc[i, j] == 1 and 
                    self.strong_relation_matrix.iloc[j, i] == 1):
                    mutual_relations.append((self.factors[i], self.factors[j]))
        
        print(f"识别到 {len(mutual_relations)} 对互为强因果关系的因素:")
        for pair in mutual_relations:
            print(f"- {pair[0]} ↔ {pair[1]}")
        
        return mutual_relations
    
    def find_causal_loops(self, max_loop_length=4):
        """
        识别因果回路
        
        参数:
        max_loop_length: 最大回路长度
        """
        if self.strong_relation_matrix is None:
            print("错误: 请先构建强关联矩阵!")
            return []
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for factor in self.factors:
            G.add_node(factor)
        
        # 添加边（基于强关联矩阵）
        for i in range(len(self.factors)):
            for j in range(len(self.factors)):
                if self.strong_relation_matrix.iloc[i, j] == 1:
                    G.add_edge(self.factors[i], self.factors[j])
        
        # 寻找所有简单回路
        all_cycles = []
        
        try:
            # 使用NetworkX寻找所有简单回路
            cycles = list(nx.simple_cycles(G))
            
            # 按长度筛选回路
            for cycle in cycles:
                if len(cycle) <= max_loop_length:
                    all_cycles.append(cycle)
            
            # 按回路长度排序
            all_cycles.sort(key=len)
            
            print(f"发现 {len(all_cycles)} 个因果回路:")
            for i, cycle in enumerate(all_cycles, 1):
                cycle_str = " → ".join(cycle) + " → " + cycle[0]
                print(f"回路{i} (长度{len(cycle)}): {cycle_str}")
            
        except Exception as e:
            print(f"寻找因果回路时出错: {e}")
        
        return all_cycles
    
    def analyze_loop_importance(self, loops):
        """
        分析回路重要性
        基于回路中因素的D+R值和影响强度
        """
        if not loops:
            return []
        
        loop_importance = []
        
        for loop in loops:
            # 计算回路中因素的平均重要性(D+R)
            loop_factors_importance = []
            for factor in loop:
                if factor in self.dr_data.index:
                    importance = self.dr_data.loc[factor, 'D+R']
                    loop_factors_importance.append(importance)
            
            avg_importance = np.mean(loop_factors_importance) if loop_factors_importance else 0
            
            # 计算回路的总影响强度
            total_influence = 0
            for i in range(len(loop)):
                current_factor = loop[i]
                next_factor = loop[(i + 1) % len(loop)]
                if current_factor in self.tc_data.index and next_factor in self.tc_data.columns:
                    influence = self.tc_data.loc[current_factor, next_factor]
                    total_influence += influence
            
            loop_importance.append({
                'loop': loop,
                'avg_importance': avg_importance,
                'total_influence': total_influence,
                'length': len(loop)
            })
        
        # 按重要性排序
        loop_importance.sort(key=lambda x: x['avg_importance'] * x['total_influence'], reverse=True)
        
        return loop_importance
    
    def visualize_causal_network(self, save_path=None):
        """可视化因果网络图"""
        if self.strong_relation_matrix is None:
            print("错误: 请先构建强关联矩阵!")
            return

        # 创建图形 - 调整为正方形
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # === 强关联矩阵热力图 ===
        # 使用专业的学术配色方案
        sns.heatmap(self.strong_relation_matrix.astype(int),
                   annot=True,
                   cmap='Blues',  # 使用蓝色系，更适合学术论文
                   center=0,
                   fmt='d',
                   ax=ax,
                   cbar=False,  # 去掉颜色条
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'},  # 增大矩阵数字字体
                   linewidths=0.5,  # 添加网格线
                   linecolor='white')  # 网格线颜色

        # 不设置标题
        ax.set_xlabel('影响的因素', fontsize=20, fontweight='bold')  # 增大轴标签字体
        ax.set_ylabel('被影响的因素', fontsize=20, fontweight='bold')  # 增大轴标签字体

        # 调整刻度标签
        ax.tick_params(axis='both', which='major', labelsize=15)  # 增大刻度标签字体
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # === 在热力图上标注互为强因果关系的连线 ===
        # 识别互为强因果关系的因素对
        mutual_relations = []
        for i in range(len(self.factors)):
            for j in range(i+1, len(self.factors)):  # 避免重复检查
                if (self.strong_relation_matrix.iloc[i, j] == 1 and
                    self.strong_relation_matrix.iloc[j, i] == 1):
                    mutual_relations.append((i, j, self.factors[i], self.factors[j]))

        # 在热力图上画箭头表示互为强因果关系
        for i, j, _, _ in mutual_relations:
            # 计算格子中心坐标
            x1, y1 = j + 0.5, i + 0.5  # (i,j)格子中心
            x2, y2 = i + 0.5, j + 0.5  # (j,i)格子中心

            # 画双向箭头 - 使用更专业的颜色
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='<->', color='darkred', lw=3, alpha=0.9))

            # 在连线中点添加标签
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, '↔', ha='center', va='center',
                   fontsize=18, fontweight='bold', color='darkred',  # 增大箭头字体
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='darkred'))

        # === 在图的底部添加说明信息 ===
        # 左下角：红线图示 + 说明
        # 绘制红色双向箭头图示
        arrow_y = 0.04  # 箭头的垂直位置
        arrow_start_x = 0.12  # 箭头起始位置
        arrow_end_x = 0.17  # 箭头结束位置

        # 使用annotate绘制双向箭头
        fig.add_artist(plt.Line2D([arrow_start_x, arrow_end_x], [arrow_y, arrow_y],
                                  transform=fig.transFigure,
                                  color='darkred', linewidth=3, alpha=0.9))
        # 添加箭头头部
        # fig.add_artist(plt.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
        #                            xycoords='figure fraction', textcoords='figure fraction',
        #                            arrowprops=dict(arrowstyle='<->', color='darkred', lw=3, alpha=0.9)))

        # 红线说明文字
        left_text = f'表示互为强因果关系 ({len(mutual_relations)}对)'
        fig.text(0.18, arrow_y, left_text,
                ha='left', va='center',
                fontsize=18,
                fontweight='normal',
                transform=fig.transFigure)

        # 右下角：阈值说明
        right_text = f'阈值λ = {self.threshold:.4f}'
        fig.text(0.8, arrow_y, right_text,
                ha='right', va='center',
                fontsize=18,
                fontweight='normal',
                transform=fig.transFigure)

        # 调整布局以确保底部说明完全显示，并与X轴标签分开
        plt.subplots_adjust(bottom=0.12)  # 为底部说明预留更多空间

        if save_path:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"因果网络图已保存至: {save_path}")

        plt.show()
    
    def save_results(self, output_dir):
        """保存分析结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存强关联矩阵
        strong_relation_file = os.path.join(output_dir, 'strong_relation_matrix.xlsx')
        self.strong_relation_matrix.to_excel(strong_relation_file)
        print(f"强关联矩阵已保存至: {strong_relation_file}")
        
        # 保存分析报告
        report_file = os.path.join(output_dir, 'causal_loop_analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DEMATEL因果回路分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据来源: {self.data_dir}\n")
            f.write(f"因素数量: {len(self.factors)}\n")
            f.write(f"设定阈值λ: {self.threshold:.4f}\n\n")
            
            # 强关联统计
            strong_count = np.sum(self.strong_relation_matrix.values)
            total_count = len(self.factors) * (len(self.factors) - 1)
            f.write(f"强关联关系统计:\n")
            f.write(f"- 强关联关系数量: {strong_count}\n")
            f.write(f"- 总可能关系数量: {total_count}\n")
            f.write(f"- 强关联比例: {strong_count/total_count:.2%}\n\n")
            
            # 互为强因果关系的因素对
            mutual_relations = self.identify_mutual_strong_relations()
            f.write(f"互为强因果关系的因素对 ({len(mutual_relations)}对):\n")
            for pair in mutual_relations:
                f.write(f"- {pair[0]} ↔ {pair[1]}\n")
            f.write("\n")
            
            # 因果回路
            loops = self.find_causal_loops()
            f.write(f"识别的因果回路 ({len(loops)}个):\n")
            for i, loop in enumerate(loops, 1):
                cycle_str = " → ".join(loop) + " → " + loop[0]
                f.write(f"回路{i} (长度{len(loop)}): {cycle_str}\n")
            
            # 回路重要性分析
            if loops:
                f.write("\n关键回路重要性排序:\n")
                loop_importance = self.analyze_loop_importance(loops)
                for i, loop_info in enumerate(loop_importance, 1):
                    loop = loop_info['loop']
                    cycle_str = " → ".join(loop) + " → " + loop[0]
                    f.write(f"第{i}重要回路: {cycle_str}\n")
                    f.write(f"  - 平均重要性: {loop_info['avg_importance']:.4f}\n")
                    f.write(f"  - 总影响强度: {loop_info['total_influence']:.4f}\n")
                    f.write(f"  - 回路长度: {loop_info['length']}\n\n")
        
        print(f"分析报告已保存至: {report_file}")


def main():
    """主函数"""
    # 数据目录
    data_dir = "result/20251026_175633"
    
    # 输出目录
    output_dir = os.path.join(data_dir, "causal_loop_analysis")
    
    print("DEMATEL因果回路分析")
    print("=" * 50)
    
    # 创建分析器
    analyzer = CausalLoopAnalyzer(data_dir)
    
    # 加载数据
    if not analyzer.load_data():
        print("数据加载失败，程序退出")
        return
    
    # 设置阈值（可以尝试不同方法）
    print("\n1. 计算阈值...")
    threshold_methods = [
        ('percentile', 90),
        ('percentile', 70),
        ('percentile', 60),
        ('mean', None),
        ('median', None)
    ]
    
    best_threshold = None
    best_method = None
    
    for method, param in threshold_methods:
        if method == 'percentile':
            threshold = analyzer.calculate_threshold(method=method, percentile=param)
        else:
            threshold = analyzer.calculate_threshold(method=method)
        
        # 构建强关联矩阵
        analyzer.build_strong_relation_matrix()
        
        # 检查强关联数量是否合理
        strong_count = np.sum(analyzer.strong_relation_matrix.values)
        total_count = len(analyzer.factors) * (len(analyzer.factors) - 1)
        ratio = strong_count / total_count
        
        print(f"方法{method}({param if param else ''}): 阈值={threshold:.4f}, 强关联比例={ratio:.2%}")
        
        # 选择合理的阈值（强关联比例在10%-30%之间）
        if 0.1 <= ratio <= 0.3:
            best_threshold = threshold
            best_method = (method, param)
            break
    
    # 如果没有找到合理的阈值，使用70%分位数
    if best_threshold is None:
        best_threshold = analyzer.calculate_threshold(method='percentile', percentile=70)
        best_method = ('percentile', 70)
    
    print(f"\n选择最佳阈值: {best_threshold:.4f} (方法: {best_method[0]})")
    
    # 构建强关联矩阵
    print("\n2. 构建强关联矩阵...")
    analyzer.build_strong_relation_matrix()
    
    # 识别互为强因果关系的因素对
    print("\n3. 识别互为强因果关系...")
    mutual_relations = analyzer.identify_mutual_strong_relations()
    
    # 寻找因果回路
    print("\n4. 寻找因果回路...")
    loops = analyzer.find_causal_loops(max_loop_length=5)
    
    # 分析回路重要性
    if loops:
        print("\n5. 分析回路重要性...")
        loop_importance = analyzer.analyze_loop_importance(loops)
        
        print("\n关键回路排序:")
        for i, loop_info in enumerate(loop_importance[:5], 1):  # 显示前5个重要回路
            loop = loop_info['loop']
            cycle_str = " → ".join(loop) + " → " + loop[0]
            print(f"第{i}重要回路: {cycle_str}")
            print(f"  平均重要性: {loop_info['avg_importance']:.4f}")
            print(f"  总影响强度: {loop_info['total_influence']:.4f}")
    
    # 可视化
    print("\n6. 生成可视化图表...")
    viz_path = os.path.join(output_dir, 'causal_loop_network.png')
    analyzer.visualize_causal_network(save_path=viz_path)
    
    # 保存结果
    print("\n7. 保存分析结果...")
    analyzer.save_results(output_dir)
    
    print(f"\n分析完成！结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()