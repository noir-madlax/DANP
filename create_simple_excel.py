import pandas as pd
import numpy as np
import os

# 创建示例数据
print("正在创建简单的专家评估Excel文件...")

# 准则名称
criteria = ['c11', 'c12', 'c21', 'c22']
headers = ['Unnamed: 0'] + criteria

# 创建Excel文件
with pd.ExcelWriter('simple_experts.xlsx') as writer:
    # 创建names工作表
    names_df = pd.DataFrame({'names': [None]*5, 'expert No': range(1,6)})
    names_df.to_excel(writer, sheet_name='names', index=False)
    print("已创建names工作表")
    
    # 创建专家工作表
    for i in range(1, 6):
        # 创建4x5矩阵，第一列是标识
        data = []
        for j, name in enumerate(criteria):
            row = [name]  # 第一列是准则名称
            # 添加随机评分(0-4)
            row.extend(np.random.randint(0, 5, size=4))
            data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        df.to_excel(writer, sheet_name=f'expert {i}', index=False)
        print(f"已创建expert {i}工作表")

print("已创建simple_experts.xlsx文件")
print(f"文件位置: {os.path.abspath('simple_experts.xlsx')}") 