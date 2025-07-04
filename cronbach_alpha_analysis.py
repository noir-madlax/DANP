import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

def calculate_cronbach_alpha_manual(data_matrix):
    """
    手动计算Cronbach's α系数
    
    参数:
    data_matrix (numpy.ndarray): 数据矩阵，行为专家，列为项目
    
    返回:
    float: Cronbach's α系数
    """
    try:
        # 确保是numpy数组
        if isinstance(data_matrix, pd.DataFrame):
            data_matrix = data_matrix.values
        
        n_experts, n_items = data_matrix.shape
        
        if n_experts < 2 or n_items < 2:
            return None
        
        # 计算每个项目的方差
        item_variances = np.var(data_matrix, axis=0, ddof=1)
        
        # 计算总分
        total_scores = np.sum(data_matrix, axis=1)
        
        # 计算总分方差
        total_variance = np.var(total_scores, ddof=1)
        
        # 计算Cronbach's α
        sum_item_variances = np.sum(item_variances)
        alpha = (n_items / (n_items - 1)) * (1 - sum_item_variances / total_variance)
        
        return alpha
    
    except Exception as e:
        print(f"计算Cronbach's α时出错: {e}")
        return None

def calculate_citc(data_matrix, item_index):
    """
    计算校正后项目与总分相关性 (CITC)
    
    参数:
    data_matrix (numpy.ndarray): 数据矩阵
    item_index (int): 项目索引
    
    返回:
    float: CITC值
    """
    try:
        if isinstance(data_matrix, pd.DataFrame):
            data_matrix = data_matrix.values
        
        # 获取该项目的得分
        item_scores = data_matrix[:, item_index]
        
        # 计算总分（排除该项目）
        other_items = np.delete(data_matrix, item_index, axis=1)
        total_scores_without_item = np.sum(other_items, axis=1)
        
        # 计算相关系数
        correlation, _ = pearsonr(item_scores, total_scores_without_item)
        
        return correlation
    
    except Exception as e:
        print(f"计算CITC时出错: {e}")
        return None

def calculate_cronbach_alpha(data_df, source_description=""):
    """
    为给定的 DataFrame 计算克隆巴赫 alpha 系数。

    参数:
    data_df (pd.DataFrame): 包含量表项目数据的 DataFrame。
                           每一行代表一个被试/观察，每一列代表一个项目。
    source_description (str): 用于日志记录的数据来源描述。

    返回:
    float: 克隆巴赫 alpha 系数，如果计算成功。
    None: 如果数据为空或无法计算。
    """
    try:
        if data_df.empty:
            print(f"信息 ({source_description}): 提供的数据框为空。")
            return None

        # 确保数据都是数值类型，实际上传入前就应处理好，这里作为保障
        data_for_alpha = data_df.select_dtypes(include=['number'])
        
        if data_for_alpha.shape[1] == 0: # 没有数值列
             print(f"错误 ({source_description}): 在提供的数据中未找到任何数值类型的列用于计算。")
             return None
        
        if data_for_alpha.shape[1] < data_df.shape[1]:
            print(f"警告 ({source_description}): 数据中包含非数值列，这些列将不会用于计算 Alpha。仅使用 {data_for_alpha.shape[1]} 个数值列。")

        # 删除任何包含 NaN 值的行 (即，如果某个专家对任何一个项目的评分缺失，则该专家的数据不用于计算)
        # 或者，如果某列（项目）对所有专家都包含NaN，也会影响
        data_for_alpha = data_for_alpha.dropna()

        if data_for_alpha.shape[0] < 2 or data_for_alpha.shape[1] < 2:
            print(f"错误 ({source_description}): 清理缺失值后，用于计算 Alpha 的数据行数 (被试/专家数: {data_for_alpha.shape[0]}) 或列数 (项目/影响关系数: {data_for_alpha.shape[1]}) 不足。至少需要2行2列。")
            return None

        # 使用手动计算方法
        alpha_value = calculate_cronbach_alpha_manual(data_for_alpha)
        
        return alpha_value

    except Exception as e:
        print(f"错误 ({source_description}): 计算克隆巴赫 alpha 时发生错误：{e}")
        return None

def perform_item_analysis(data_df, item_names):
    """
    执行逐项分析
    
    参数:
    data_df (pd.DataFrame): 数据框
    item_names (list): 项目名称列表
    
    返回:
    list: 分析结果列表
    """
    try:
        # 计算整体α系数
        overall_alpha = calculate_cronbach_alpha_manual(data_df)
        
        if overall_alpha is None:
            print("无法计算整体Cronbach's α")
            return None
        
        print(f"\n整体 Cronbach's α = {overall_alpha:.4f}")
        
        # 信度解释
        if overall_alpha >= 0.9:
            interpretation = "极好 (Excellent)"
        elif overall_alpha >= 0.8:
            interpretation = "很好 (Good)"
        elif overall_alpha >= 0.7:
            interpretation = "可接受 (Acceptable)"
        elif overall_alpha >= 0.6:
            interpretation = "有问题 (Questionable)"
        else:
            interpretation = "不可接受 (Unacceptable)"
        
        print(f"信度解释: {interpretation}")
        
        # 逐项分析
        print("\n逐项分析...")
        print("=" * 100)
        print(f"{'项目名称':<30} {'CITC':<8} {'删除后α':<10} {'α变化':<8} {'建议':<15}")
        print("=" * 100)
        
        analysis_results = []
        data_matrix = data_df.values
        
        for i, item_name in enumerate(item_names):
            # 计算CITC
            citc = calculate_citc(data_matrix, i)
            
            # 计算删除该项目后的α系数
            data_without_item = np.delete(data_matrix, i, axis=1)
            alpha_without = calculate_cronbach_alpha_manual(data_without_item)
            
            # 计算α变化
            alpha_change = alpha_without - overall_alpha if alpha_without is not None else None
            
            # 生成建议
            if citc is not None and citc < 0.3:
                suggestion = "考虑删除"
            elif alpha_change is not None and alpha_change > 0.01:
                suggestion = "删除后提升"
            elif citc is not None and citc > 0.5:
                suggestion = "保留"
            else:
                suggestion = "需要关注"
            
            # 显示结果
            citc_str = f"{citc:.4f}" if citc is not None else "N/A"
            alpha_without_str = f"{alpha_without:.4f}" if alpha_without is not None else "N/A"
            change_str = f"{alpha_change:+.4f}" if alpha_change is not None else "N/A"
            
            # 截断过长的项目名称以便显示
            display_name = item_name[:28] + ".." if len(item_name) > 30 else item_name
            print(f"{display_name:<30} {citc_str:<8} {alpha_without_str:<10} {change_str:<8} {suggestion:<15}")
            
            # 保存结果
            analysis_results.append({
                '项目名称': item_name,
                'CITC': citc,
                '删除后α系数': alpha_without,
                'α系数变化': alpha_change,
                '建议': suggestion
            })
        
        print("=" * 100)
        
        # 统计分析
        low_citc_count = sum(1 for r in analysis_results if r['CITC'] is not None and r['CITC'] < 0.3)
        high_improvement_count = sum(1 for r in analysis_results if r['α系数变化'] is not None and r['α系数变化'] > 0.01)
        
        print(f"\n分析统计:")
        print(f"• 低CITC项目 (< 0.3): {low_citc_count} 个")
        print(f"• 删除后显著提升信度的项目 (Δα > 0.01): {high_improvement_count} 个")
        
        if low_citc_count > 0 or high_improvement_count > 0:
            print("\n建议关注的项目:")
            for result in analysis_results:
                if ((result['CITC'] is not None and result['CITC'] < 0.3) or 
                    (result['α系数变化'] is not None and result['α系数变化'] > 0.01)):
                    print(f"  • {result['项目名称']}: {result['建议']}")
        
        return analysis_results, overall_alpha, interpretation
        
    except Exception as e:
        print(f"执行逐项分析时出错: {e}")
        return None

if __name__ == "__main__":
    excel_file_path = "ALL.xlsx"
    all_experts_flattened_data = []
    item_names = None # 用于存储扁平化后的项目名称（例如 C1_influences_C1）
    expected_num_items = -1

    try:
        xls = pd.ExcelFile(excel_file_path)
        sheet_names = xls.sheet_names

        if not sheet_names or len(sheet_names) <= 1:
            print(f"错误或警告：Excel 文件 '{excel_file_path}' 没有足够的工作表用于分析（至少需要一个专家名单表和一个数据表）。")
        else:
            first_sheet_name = sheet_names[0]
            expert_questionnaire_sheets = sheet_names[1:]
            
            print(f"总共找到 {len(sheet_names)} 个工作表: {sheet_names}")
            print(f"将跳过第一个工作表: '{first_sheet_name}' (专家名单)。")
            print(f"将尝试汇总以下 {len(expert_questionnaire_sheets)} 个专家问卷工作表的数据: {expert_questionnaire_sheets}\n")

            for sheet_name in expert_questionnaire_sheets:
                print(f"--- 正在读取专家问卷: '{sheet_name}' ---")
                try:
                    df_expert_matrix = pd.read_excel(xls, sheet_name=sheet_name, header=0, index_col=0)
                    
                    # 确保矩阵不为空
                    if df_expert_matrix.empty:
                        print(f"警告 (工作表: {sheet_name}): 此工作表为空，将跳过。")
                        continue
                    
                    # 选择数值部分进行扁平化
                    numeric_matrix_part = df_expert_matrix.select_dtypes(include=['number'])
                    if numeric_matrix_part.empty or numeric_matrix_part.shape[1] == 0:
                        print(f"警告 (工作表: {sheet_name}): 在此工作表中未找到有效的数值数据区域，将跳过。")
                        continue

                    # 排除对角线元素进行扁平化
                    flattened_expert_data = []
                    temp_item_names_for_this_expert = []
                    
                    for r_idx, r_name in enumerate(numeric_matrix_part.index):
                        for c_idx, c_name in enumerate(numeric_matrix_part.columns):
                            # 排除对角线元素（自己影响自己）
                            if r_idx != c_idx:
                                value = numeric_matrix_part.iloc[r_idx, c_idx]
                                flattened_expert_data.append(value)
                                temp_item_names_for_this_expert.append(f"{str(r_name).strip()}_influences_{str(c_name).strip()}")
                    
                    current_num_items_from_sheet = len(flattened_expert_data)

                    if item_names is None: # 这是第一个成功处理的专家工作表
                        expected_num_items = current_num_items_from_sheet
                        item_names = temp_item_names_for_this_expert
                        
                        print(f"信息: 从工作表 '{sheet_name}' 初始化了 {expected_num_items} 个影响关系项目（排除对角线）。")

                    elif current_num_items_from_sheet != expected_num_items:
                        print(f"警告 (工作表: {sheet_name}): 此工作表包含 {current_num_items_from_sheet} 个数据点，与预期的 {expected_num_items} 个不符。将跳过此工作表以保持数据一致性。")
                        continue
                    
                    all_experts_flattened_data.append(flattened_expert_data)
                    print(f"信息: 已成功处理工作表 '{sheet_name}'，包含 {current_num_items_from_sheet} 个数据点（排除对角线）。")

                except Exception as e:
                    print(f"处理工作表 '{sheet_name}' 时发生错误: {e}。将跳过此工作表。")
                finally:
                    print("-" * 40)
            
            if not all_experts_flattened_data or item_names is None:
                print("\n未能收集到任何有效且一致的专家数据用于最终分析。")
            else:
                print(f"\n--- 所有有效专家问卷已汇总 ({len(all_experts_flattened_data)}份问卷) --- ")
                # 创建汇总的 DataFrame
                aggregated_df = pd.DataFrame(all_experts_flattened_data, columns=item_names)
                print(f"汇总后的数据表维度: {aggregated_df.shape[0]} 位专家, {aggregated_df.shape[1]} 个影响关系项目（排除对角线）。")

                print("\n--- 正在对所有专家的汇总数据进行克隆巴赫 Alpha 分析 ---")
                
                # 执行完整的逐项分析
                analysis_results = perform_item_analysis(aggregated_df, item_names)
                
                if analysis_results is not None:
                    results_list, overall_alpha, interpretation = analysis_results
                    
                    print(f"\n最终克隆巴赫 Alpha (α) 系数: {overall_alpha:.4f}")
                    print(f"信度解释: {interpretation}")
                    
                    # 保存结果到Excel
                    print("\n保存分析结果...")
                    results_df = pd.DataFrame(results_list)
                    
                    # 添加总体信息
                    summary_info = pd.DataFrame({
                        '分析项目': ['整体Cronbach\'s α', '信度解释', '专家数量', '项目数量（排除对角线）'],
                        '结果': [f"{overall_alpha:.4f}", interpretation, len(all_experts_flattened_data), len(item_names)]
                    })
                    
                    output_file = 'cronbach_alpha_detailed_analysis.xlsx'
                    
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        # 逐项分析结果
                        results_df.to_excel(writer, sheet_name='逐项分析', index=False)
                        
                        # 总体信息
                        summary_info.to_excel(writer, sheet_name='总体信息', index=False)
                        
                        # 原始汇总数据
                        aggregated_df.to_excel(writer, sheet_name='汇总数据', index=True)
                    
                    print(f"详细分析结果已保存到: {output_file}")
                else:
                    print("\n未能完成逐项分析。")

    except FileNotFoundError:
        print(f"错误：文件 '{excel_file_path}' 未找到。请确保文件与脚本在同一目录，或提供完整路径。")
    except ImportError:
        print("错误：缺少必要的库。请确保已安装 pandas、numpy、scipy 和 openpyxl。")
        print("您可以使用以下命令安装：pip install pandas numpy scipy openpyxl")
    except Exception as e:
        print(f"处理 Excel 文件或执行分析时发生未预料的错误：{e}")
        import traceback
        traceback.print_exc() 