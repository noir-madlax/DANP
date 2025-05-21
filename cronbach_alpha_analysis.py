import pandas as pd
import pingouin as pg

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

        # 计算克隆巴赫 alpha
        alpha_results = pg.cronbach_alpha(data=data_for_alpha)
        
        return alpha_results[0] # 返回 alpha 值

    except Exception as e:
        print(f"错误 ({source_description}): 计算克隆巴赫 alpha 时发生错误：{e}")
        return None

if __name__ == "__main__":
    excel_file_path = "paper_data.xlsx"
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

                    flattened_expert_data = numeric_matrix_part.values.flatten().tolist()
                    current_num_items_from_sheet = len(flattened_expert_data)

                    if item_names is None: # 这是第一个成功处理的专家工作表
                        expected_num_items = current_num_items_from_sheet
                        # 生成项目名称 (基于第一个有效工作表的行列名)
                        temp_item_names = []
                        for r_name in numeric_matrix_part.index:
                            for c_name in numeric_matrix_part.columns:
                                temp_item_names.append(f"{str(r_name).strip()}_influences_{str(c_name).strip()}")
                        item_names = temp_item_names
                        
                        if len(item_names) != expected_num_items:
                            print(f"错误 (工作表: {sheet_name}): 自动生成的项目名称数量 ({len(item_names)}) 与扁平化数据点数量 ({expected_num_items}) 不匹配。请检查矩阵结构和脚本逻辑。")
                            item_names = None # 重置以避免后续错误
                            expected_num_items = -1
                            continue # 跳过这个有问题的表
                        print(f"信息: 从工作表 '{sheet_name}' 初始化了 {expected_num_items} 个影响关系项目。")

                    elif current_num_items_from_sheet != expected_num_items:
                        print(f"警告 (工作表: {sheet_name}): 此工作表包含 {current_num_items_from_sheet} 个数据点，与预期的 {expected_num_items} 个不符。将跳过此工作表以保持数据一致性。")
                        continue
                    
                    all_experts_flattened_data.append(flattened_expert_data)
                    print(f"信息: 已成功处理工作表 '{sheet_name}'，包含 {current_num_items_from_sheet} 个数据点。")

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
                print(f"汇总后的数据表维度: {aggregated_df.shape[0]} 位专家, {aggregated_df.shape[1]} 个影响关系项目。")
                # print("汇总数据的前几行和项目名:")
                # print(aggregated_df.head())
                # print(f"项目名: {aggregated_df.columns.tolist()}")

                print("\n--- 正在对所有专家的汇总数据进行克隆巴赫 Alpha 分析 ---")
                final_alpha = calculate_cronbach_alpha(aggregated_df, source_description="汇总数据 (所有专家)")

                if final_alpha is not None:
                    print(f"\n最终克隆巴赫 Alpha (α) 系数 (基于所有专家汇总数据): {final_alpha:.4f}")
                    if final_alpha >= 0.9:
                        print("解释：信度极好 (Excellent)")
                    elif final_alpha >= 0.8:
                        print("解释：信度很好 (Good)")
                    elif final_alpha >= 0.7:
                        print("解释：信度可以接受 (Acceptable)")
                    elif final_alpha >= 0.6:
                        print("解释：信度有问题/值得怀疑 (Questionable)")
                    elif final_alpha >= 0.5:
                        print("解释：信度较差 (Poor)")
                    else:
                        print("解释：信度不可接受 (Unacceptable)")
                else:
                    print("\n未能计算最终的克隆巴赫 Alpha 系数。")

    except FileNotFoundError:
        print(f"错误：文件 '{excel_file_path}' 未找到。请确保文件与脚本在同一目录，或提供完整路径。")
    except ImportError:
        print("错误：缺少必要的库。请确保已安装 pandas 和 pingouin。")
        print("您可以使用以下命令安装：pip install pandas pingouin openpyxl")
    except Exception as e:
        print(f"处理 Excel 文件或执行分析时发生未预料的错误：{e}") 