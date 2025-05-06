import pandas as pd
import os
import datetime

# 全局变量用于存储当前运行的目录ID
run_id = None

import pandas as pd


def frame_list(e_count):
    """ creating a list of data frames based on experts count"""
    df = [None] * e_count
    i = 1
    while i <= e_count:
        df[i - 1] = pd.read_excel(r'my_data.xlsx', sheet_name=f'expert {i}')
        i += 1
    return df


def listing(e_count, df, form = 1):
    """this function creates a list out of data frames

    we created a sublist to append all rows to a single list and then add
    that sublist to my_list. my_list is a list of each expert questionnaire
    values categorized by row. each row's values are then converted to
    their fuzzy state
    """
    my_list = [None] * e_count
    # this None list is to prevent errors when trying to put sublist in it
    j = 0
    while j < e_count:
        sublist = []
        j += 1
        count_row = df[j - 1].shape[0]
        for i in range(0, count_row):
            row_list = df[j - 1].iloc[i].tolist()
            # iloc[i] separates row i and del index 0 is for removing c11 & ....
            del row_list[0]
            # fuzzification
            if form == 1:
                row_list = fuzzification(row_list)
            sublist.append(row_list)
        my_list[j - 1] = sublist
    return my_list


def fuzzification(lst):
    """ converts absolute numbers to a triangular fuzzy set of numbers"""
    lst = [[0, 0, 0.25] if i == 0 else
           [0, 0.25, 0.5] if i == 1 else
           [0.25, 0.5, 0.75] if i == 2 else
           [0.5, 0.75, 1] if i == 3 else
           [0.75, 1, 1] if i == 4 else
           i for i in lst]
    return lst


def defuzzification(count, T_a, T_b, T_c):
    super_matrix = []
    for i in range(0, int(count)):
        super_matrix2 = []
        super_matrix.append(super_matrix2)
        for j in range(0, int(count)):
            super_matrix2.append(
                (((T_c[i, j] - T_a[i, j]) + (T_b[i, j] - T_a[i, j])) / 3)
                + T_a[i, j])
    return super_matrix


def sum_mean(e_count, mylist):
    """sums all experts opinions into a single list and calculates their mean"""
    final_list = []
    for i in range(0, len(mylist[0])):
        # i&j for enough steps(list in lists)
        sublist2 = []
        final_list.append(sublist2)
        for j in range(0, len(mylist[0])):
            summation = [sum(z) for z in zip(mylist[0][i][j], mylist[1][i][j])]
            # it is safe assume that we have opinions from at least 2 experts
            # if there more than 2 experts the following loop will address them
            k = 2
            while k < e_count:
                summation = [sum(z) for z in zip(summation, mylist[k][i][j])]
                k += 1
            # here the sum of all experts for a particular question is divided
            # to acquire mean
            mean = [sum(z) / e_count for z in zip(summation)]
            sublist2.append(mean)
    return (final_list)


def exl_out(list, name, ind=None, col=None):
    """creating excel output"""
    global run_id
    
    # 创建result目录（如果不存在）
    if not os.path.exists('result'):
        os.makedirs('result')
    
    # 如果这是第一次调用，创建一个唯一的运行ID
    if run_id is None:
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('result', run_id)
        os.makedirs(result_dir)
        print(f"所有结果文件将保存在目录: {result_dir}")
    else:
        result_dir = os.path.join('result', run_id)
    
    # 创建DataFrame
    if ind is not None and col is not None:
        dff = pd.DataFrame(list, index=ind, columns=col)
    elif ind is not None:
        dff = pd.DataFrame(list, index=ind)
    elif col is not None:
        dff = pd.DataFrame(list, columns=col)
    else:
        dff = pd.DataFrame(list)
    
    # 保存到目标目录
    file_path = os.path.join(result_dir, f'{name}.xlsx')
    dff.to_excel(file_path)
