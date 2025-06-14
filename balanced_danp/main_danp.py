import numpy as np
import inputs as ip
import data_organizer as do
import computation as cp
import warnings


# just to get rid of numpy array type error
warnings.filterwarnings("ignore", category=DeprecationWarning)

# getting the count of experts
e_count = ip.exp_count()

# creating data frame
df = do.frame_list(e_count)

# making a list that contains all sheets row by row
mylist = do.listing(e_count, df)

# sum and mean of all experts opinions
final_list = do.sum_mean(e_count, mylist)

# first step of fuzzy DANP: creating arrays for l,m,u
X_a = np.zeros((len(final_list), len(final_list)))
X_b = np.zeros((len(final_list), len(final_list)))
X_c = np.zeros((len(final_list), len(final_list)))

# replacing with actual value
for i in range(0, len(final_list)):
    for j in range(0, len(final_list)):
        a, b, c = final_list[i][j]
        X_a[i, j] = a
        X_b[i, j] = b
        X_c[i, j] = c

# normalizing, axis 0 for column sum, 1 for row sum
X_a = X_a / np.max(np.sum(X_a, axis=1))
X_b = X_b / np.max(np.sum(X_b, axis=1))
X_c = X_c / np.max(np.sum(X_c, axis=1))

# calculating identity matrix minus x and inverting it
Y_a = np.linalg.inv(np.identity(len(final_list)) - X_a)
Y_b = np.linalg.inv(np.identity(len(final_list)) - X_b)
Y_c = np.linalg.inv(np.identity(len(final_list)) - X_c)

# calculating total relation matrix
T_a = np.matmul(X_a, Y_a)
T_b = np.matmul(X_b, Y_b)
T_c = np.matmul(X_c, Y_c)

# calculating R, D & ...
D_a = np.sum(T_a, axis=1)
D_b = np.sum(T_b, axis=1)
D_c = np.sum(T_c, axis=1)
R_a = np.sum(T_a, axis=0)
R_b = np.sum(T_b, axis=0)
R_c = np.sum(T_c, axis=0)
defuzzied_D = (((D_c - D_a) + (D_b - D_a)) / 3) + D_a
defuzzied_R = (((R_c - R_a) + (R_b - R_a)) / 3) + R_a
D_plus_R = defuzzied_D + defuzzied_R
D_minus_R = defuzzied_D - defuzzied_R

# getting the count of dimensions and criteria
dimension_count = ip.dim_count()
criteria_count = ip.crit_count(dimension_count)
criteria_names = ip.crit_name(dimension_count, criteria_count)

# defuzzification & outputting the DEMATEL Tc final matrix
supermatrix = do.defuzzification(len(criteria_names), T_a, T_b, T_c)

do.exl_out(supermatrix, 'Tc_defuzzied', criteria_names, criteria_names)

# defining a dictionary for a table presenting R, D & ...
data2 = {'D_l': D_a, 'D_m': D_b, 'D_u': D_c, 'R_l': R_a, 'R_m': R_b, 'R_u': R_c,
         'defuzzied_D': defuzzied_D, 'defuzzied_R': defuzzied_R, 'D+R': D_plus_R,
         'D-R': D_minus_R}

do.exl_out(data2, 'D_R', criteria_names)

mylist_l = []
for i in range(0, len(mylist[0])):
    sublist_u = []
    mylist_l.append(sublist_u)
    for j in range(0, len(mylist[0])):
        each = final_list[i][j][0]
        sublist_u.append(each)

do.exl_out(mylist_l, 'l_experts_average', criteria_names, criteria_names)

mylist_m = []
for i in range(0, len(mylist[0])):
    sublist_m = []
    mylist_m.append(sublist_m)
    for j in range(0, len(mylist[0])):
        each = final_list[i][j][1]
        sublist_m.append(each)

do.exl_out(mylist_m, 'm_experts_average', criteria_names, criteria_names)

mylist_u = []
for i in range(0, len(mylist[0])):
    sublist_u = []
    mylist_u.append(sublist_u)
    for j in range(0, len(mylist[0])):
        each = final_list[i][j][2]
        sublist_u.append(each)

# creating output for l,m,u of experts average output
do.exl_out(mylist_u, 'u_experts_average', criteria_names, criteria_names)

# creating normalized l,m,u output
do.exl_out(X_a, 'normalized_l', criteria_names, criteria_names)
do.exl_out(X_b, 'normalized_m', criteria_names, criteria_names)
do.exl_out(X_c, 'normalized_u', criteria_names, criteria_names)

# creating total matrix l,m,u output
do.exl_out(T_a, 'Tc_l', criteria_names, criteria_names)
do.exl_out(T_b, 'Tc_m', criteria_names, criteria_names)
do.exl_out(T_c, 'Tc_u', criteria_names, criteria_names)

# lists of all matrices that will be defining our dimensional matrix or Td
# based upon our dimension count and their subset criteria
Td_a_list = cp.sub_matrices(dimension_count, criteria_count, T_a)
Td_b_list = cp.sub_matrices(dimension_count, criteria_count, T_b)
Td_c_list = cp.sub_matrices(dimension_count, criteria_count, T_c)

Td_a = cp.td_maker(dimension_count, Td_a_list)
Td_b = cp.td_maker(dimension_count, Td_b_list)
Td_c = cp.td_maker(dimension_count, Td_c_list)

dimension_names = ip.dim_name(dimension_count)

do.exl_out(Td_a, 'Td_l', dimension_names, dimension_names)
do.exl_out(Td_b, 'Td_m', dimension_names, dimension_names)
do.exl_out(Td_c, 'Td_u', dimension_names, dimension_names)

# calculating R, D for Td
Td_D_a = np.sum(Td_a, axis=1)
Td_D_b = np.sum(Td_b, axis=1)
Td_D_c = np.sum(Td_c, axis=1)
Td_R_a = np.sum(Td_a, axis=0)
Td_R_b = np.sum(Td_b, axis=0)
Td_R_c = np.sum(Td_c, axis=0)

Td_defuzzied_D = (((Td_D_c - Td_D_a) + (Td_D_b - Td_D_a)) / 3) + Td_D_a
Td_defuzzied_R = (((Td_R_c - Td_R_a) + (Td_R_b - Td_R_a)) / 3) + Td_R_a
Td_D_plus_R = Td_defuzzied_D + Td_defuzzied_R
Td_D_minus_R = Td_defuzzied_D - Td_defuzzied_R

# defining a dictionary for a table presenting R, D & ...
data3 = {'D_l': Td_D_a, 'D_m': Td_D_b, 'D_u': Td_D_c, 'R_l': Td_R_a,
         'R_m': Td_R_b, 'R_u': Td_R_c, 'defuzzied_D': Td_defuzzied_D,
         'defuzzied_R': Td_defuzzied_R, 'D+R': Td_D_plus_R, 'D-R': Td_D_minus_R}

do.exl_out(data3, 'Td_D_R', dimension_names)

# defuzzied Td
supermatrix2 = do.defuzzification(dimension_count, Td_a, Td_b, Td_c)

do.exl_out(supermatrix2, 'Td_defuzzied', dimension_names, dimension_names)

# normalizing each subset matrix on it's own
p = 0
while p < int(dimension_count) * int(dimension_count):
    Td_a_list[p] = cp.normalizing(Td_a_list[p])
    Td_b_list[p] = cp.normalizing(Td_b_list[p])
    Td_c_list[p] = cp.normalizing(Td_c_list[p])
    p += 1

# turning the list to normalized super matrix
normal_super_l = cp.t_norm_making(Td_a_list, dimension_count)
normal_super_m = cp.t_norm_making(Td_b_list, dimension_count)
normal_super_u = cp.t_norm_making(Td_c_list, dimension_count)

do.exl_out(normal_super_l, 'normal_super_l', criteria_names, criteria_names)
do.exl_out(normal_super_m, 'normal_super_m', criteria_names, criteria_names)
do.exl_out(normal_super_u, 'normal_super_u', criteria_names, criteria_names)

# Transposing super matrices to form unweighted super matrix
unweighted_l = normal_super_l.transpose()
unweighted_m = normal_super_m.transpose()
unweighted_u = normal_super_u.transpose()

do.exl_out(unweighted_l, 'unweighted_l', criteria_names, criteria_names)
do.exl_out(unweighted_m, 'unweighted_m', criteria_names, criteria_names)
do.exl_out(unweighted_u, 'unweighted_u', criteria_names, criteria_names)

# normalizing Td
Td_l_norm = cp.normalizing(Td_a)
Td_m_norm = cp.normalizing(Td_b)
Td_u_norm = cp.normalizing(Td_c)

# transposing Td
Td_l_norm_trans = Td_l_norm.transpose()
Td_m_norm_trans = Td_m_norm.transpose()
Td_u_norm_trans = Td_u_norm.transpose()

do.exl_out(Td_l_norm_trans, 'Td_l_norm_trans', dimension_names, dimension_names)
do.exl_out(Td_m_norm_trans, 'Td_m_norm_trans', dimension_names, dimension_names)
do.exl_out(Td_u_norm_trans, 'Td_u_norm_trans', dimension_names, dimension_names)

# transforming to a list of matrices in order for us to multiply them with their
# corresponding element in Td matrix
un_sup_l_list = cp.sub_matrices(dimension_count, criteria_count, unweighted_l)
un_sup_m_list = cp.sub_matrices(dimension_count, criteria_count, unweighted_m)
un_sup_u_list = cp.sub_matrices(dimension_count, criteria_count, unweighted_u)

# multiplying unweighted super with transposed normal Td to make weighted super
weighted_l_list = cp.weigh_super_list(un_sup_l_list, Td_l_norm_trans,
                                      dimension_count)
weighted_m_list = cp.weigh_super_list(un_sup_m_list, Td_m_norm_trans,
                                      dimension_count)
weighted_u_list = cp.weigh_super_list(un_sup_u_list, Td_u_norm_trans,
                                      dimension_count)

# transforming the list into super matrix
weighted_l = cp.t_norm_making(weighted_l_list, dimension_count)
weighted_m = cp.t_norm_making(weighted_m_list, dimension_count)
weighted_u = cp.t_norm_making(weighted_u_list, dimension_count)

do.exl_out(weighted_l, 'weighted_l', criteria_names, criteria_names)
do.exl_out(weighted_m, 'weighted_m', criteria_names, criteria_names)
do.exl_out(weighted_u, 'weighted_u', criteria_names, criteria_names)

# calculating limit of weighted matrix to the power of infinite odd number
limit_weighted_l = np.linalg.matrix_power(weighted_l, 101)
limit_weighted_m = np.linalg.matrix_power(weighted_m, 101)
limit_weighted_u = np.linalg.matrix_power(weighted_u, 101)

x = 0
while x < len(criteria_names):
    """ checking if we have reached to the point that matrix is stable up to 
    four decimals and there is no need for it to be raised to power anymore.
    there is a shortcoming with this code: if the conditions turn to be true
    for the next loop first elements wont be checked, surely there is little to 
    no chance for an occurrence that all the other elements but the first one 
    would be equal in a row, so i wont consider it
    """
    z = 0
    while z < (len(criteria_names) - 1):
        if limit_weighted_l[x, z] - limit_weighted_l[x, z + 1] > 0.0001:
            limit_weighted_l = np.linalg.matrix_power(weighted_l, 101)
            x, z = 0, 0
        if limit_weighted_m[x, z] - limit_weighted_m[x, z + 1] > 0.0001:
            limit_weighted_m = np.linalg.matrix_power(weighted_m, 101)
            x, z = 0, 0
        if limit_weighted_u[x, z] - limit_weighted_u[x, z + 1] > 0.0001:
            limit_weighted_u = np.linalg.matrix_power(weighted_u, 101)
            x, z = 0, 0
        z += 1
    x += 1


do.exl_out(limit_weighted_l, 'limit_weighted_l', criteria_names, criteria_names)
do.exl_out(limit_weighted_m, 'limit_weighted_m', criteria_names, criteria_names)
do.exl_out(limit_weighted_u, 'limit_weighted_u', criteria_names, criteria_names)

# 原始方法的最终权重计算
defuzzy_lim_weighted_original = ((((limit_weighted_u - limit_weighted_l) +
                         (limit_weighted_m - limit_weighted_l)) / 3)
                        + limit_weighted_l)

# ===================== 平衡权重方法开始 =====================

# 计算维度权重（基于维度内因素个数）
total_criteria = sum(criteria_count)
dimension_weights = []
for i in range(int(dimension_count)):
    dim_weight = criteria_count[i] / total_criteria
    dimension_weights.append(dim_weight)

print("维度权重分配:")
for i in range(int(dimension_count)):
    print(f"维度 {i+1} ({dimension_names[i]}): {criteria_count[i]}个因素, 权重 = {dimension_weights[i]:.4f} ({dimension_weights[i]*100:.1f}%)")

# 保存维度权重分配信息
dimension_weight_data = {
    'Dimension': dimension_names,
    'Criteria_Count': criteria_count,
    'Dimension_Weight': dimension_weights,
    'Weight_Percentage': [w*100 for w in dimension_weights]
}
do.exl_out(dimension_weight_data, 'dimension_weights_distribution', dimension_names)

# 应用维度权重到因素权重
# 首先获取每个因素原本的相对权重（在各自维度内）
defuzzy_lim_weighted_balanced = defuzzy_lim_weighted_original[:, 0].copy()

# 计算每个维度内的权重总和
criteria_start_index = 0
for dim_idx in range(int(dimension_count)):
    dim_criteria_count = criteria_count[dim_idx]
    dim_end_index = criteria_start_index + dim_criteria_count
    
    # 获取该维度内所有因素的权重
    dim_weights = defuzzy_lim_weighted_original[criteria_start_index:dim_end_index, 0]
    
    # 计算该维度内权重的总和
    dim_weight_sum = np.sum(dim_weights)
    
    # 如果维度内权重总和不为0，则重新分配
    if dim_weight_sum > 0:
        # 计算维度内各因素的相对权重（标准化到维度内）
        dim_relative_weights = dim_weights / dim_weight_sum
        
        # 应用维度权重
        balanced_dim_weights = dim_relative_weights * dimension_weights[dim_idx]
        
        # 更新平衡后的权重
        defuzzy_lim_weighted_balanced[criteria_start_index:dim_end_index] = balanced_dim_weights
    
    criteria_start_index = dim_end_index

# 输出对比结果
print("\n权重对比:")
print("因素名称\t\t原始权重\t平衡权重\t差异")
print("-" * 60)
for i in range(len(criteria_names)):
    original = defuzzy_lim_weighted_original[i, 0]
    balanced = defuzzy_lim_weighted_balanced[i]
    difference = balanced - original
    print(f"{criteria_names[i]:<15}\t{original:.6f}\t{balanced:.6f}\t{difference:+.6f}")

# 验证权重总和
print(f"\n原始权重总和: {np.sum(defuzzy_lim_weighted_original[:, 0]):.6f}")
print(f"平衡权重总和: {np.sum(defuzzy_lim_weighted_balanced):.6f}")

# 保存结果
header = ["Weights"]

# 保存原始权重
do.exl_out(defuzzy_lim_weighted_original[:, 0], 'defuzzy_lim_weighted_original', criteria_names, header)

# 保存平衡权重
do.exl_out(defuzzy_lim_weighted_balanced, 'defuzzy_lim_weighted_balanced', criteria_names, header)

# 保存权重对比表
weight_comparison = {
    'Criteria': criteria_names,
    'Original_Weight': defuzzy_lim_weighted_original[:, 0],
    'Balanced_Weight': defuzzy_lim_weighted_balanced,
    'Difference': defuzzy_lim_weighted_balanced - defuzzy_lim_weighted_original[:, 0],
    'Relative_Change_Percent': ((defuzzy_lim_weighted_balanced - defuzzy_lim_weighted_original[:, 0]) / defuzzy_lim_weighted_original[:, 0] * 100)
}
do.exl_out(weight_comparison, 'weight_comparison', criteria_names)

# ===================== 平衡权重方法结束 =====================

# 为了保持兼容性，输出默认的平衡权重作为最终结果
do.exl_out(defuzzy_lim_weighted_balanced, 'defuzzy_lim_weighted', criteria_names, header)

quit()
