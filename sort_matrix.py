import pandas as pd
import re
import os

def natural_sort_key(s):
    """对标准编号进行自然排序的函数，如C11, C12, C21会按照正确的层次结构排序"""
    # 提取C后面的数字
    match = re.match(r'C(\d+)(\d+)', s)
    if match:
        dim, crit = int(match.group(1)), int(match.group(2))
        # 返回一个元组，先按维度排序，再按标准排序
        return (dim, crit)
    return (99, 99)  # 对于不匹配的情况，放到最后

def sort_expert_matrix(file_path='paper_data.xlsx', expert_sheet='expert 1', output_path=None):
    """
    读取专家评分表并按照标准结构排序行和列
    
    参数:
    file_path: Excel文件路径
    expert_sheet: 专家表格的名称
    output_path: 输出文件路径，默认为None时生成新文件名
    
    返回:
    排序后的DataFrame
    """
    # 读取Excel表格
    df = pd.read_excel(file_path, sheet_name=expert_sheet)
    
    # 获取第一列作为标准列
    criteria_col = df.iloc[:, 0]
    
    # 创建排序索引
    sort_index = sorted(range(len(criteria_col)), key=lambda i: natural_sort_key(criteria_col[i]))
    
    # 对行进行排序
    df_sorted = df.iloc[sort_index].reset_index(drop=True)
    
    # 获取列名（假设第一行以后的列名也是C11, C12等格式）
    col_names = df_sorted.columns.tolist()
    
    # 对列进行排序（保留第一列不变）
    first_col = col_names[0]
    other_cols = col_names[1:]
    sorted_other_cols = sorted(other_cols, key=natural_sort_key)
    new_cols = [first_col] + sorted_other_cols
    
    # 重新排列列
    df_sorted = df_sorted[new_cols]
    
    # 保存排序后的表格
    if output_path is None:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_sorted{ext}")
    
    # 如果输出文件已存在，创建一个新的Excel文件
    if not os.path.exists(output_path):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_sorted.to_excel(writer, sheet_name=expert_sheet, index=False)
    else:
        # 读取现有文件
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
            df_sorted.to_excel(writer, sheet_name=expert_sheet, index=False)
    
    print(f"排序后的矩阵已保存到: {output_path}, 表格: {expert_sheet}")
    return df_sorted

def sort_all_experts(file_path='paper_data.xlsx', output_path=None):
    """排序所有专家的矩阵"""
    # 读取Excel中所有的表格名
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    
    # 为所有以"expert"开头的表格排序
    expert_sheets = [sheet for sheet in sheet_names if sheet.startswith('expert')]
    
    # 设置输出路径
    if output_path is None:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_sorted{ext}")
    
    # 删除可能存在的旧输出文件
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 排序每个专家的表格
    for sheet in expert_sheets:
        sort_expert_matrix(file_path, sheet, output_path)
    
    print(f"所有专家矩阵已排序并保存到: {output_path}")

if __name__ == "__main__":
    # 测试排序第一个专家的矩阵
    # sort_expert_matrix()
    
    # 排序所有专家的矩阵
    sort_all_experts() 