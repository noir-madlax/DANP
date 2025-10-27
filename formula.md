

## 第一阶段：数据预处理

### 1. 模糊化处理
**易懂描述：** 将专家的0-4评分转换为三角模糊数
```
0 → [0, 0, 0.25]     (无影响)
1 → [0, 0.25, 0.5]   (低影响)  
2 → [0.25, 0.5, 0.75] (中等影响)
3 → [0.5, 0.75, 1]   (高影响)
4 → [0.75, 1, 1]     (非常高影响)
```

**LaTeX表达式：**
```latex
\tilde{x}_{ij} = \begin{cases}
[0, 0, 0.25] & \text{if } x_{ij} = 0 \\
[0, 0.25, 0.5] & \text{if } x_{ij} = 1 \\
[0.25, 0.5, 0.75] & \text{if } x_{ij} = 2 \\
[0.5, 0.75, 1] & \text{if } x_{ij} = 3 \\
[0.75, 1, 1] & \text{if } x_{ij} = 4
\end{cases}
```

### 2. 专家意见聚合
**易懂描述：** 计算所有专家对每个准则对的平均评估
```
最终评估值 = (专家1评估 + 专家2评估 + ... + 专家n评估) / 专家数量
```

**LaTeX表达式：**
```latex
\tilde{x}_{ij}^{final} = \frac{1}{K} \sum_{k=1}^{K} \tilde{x}_{ij}^{k}
```
其中 $K$ 为专家数量，$\tilde{x}_{ij}^{k}$ 为第$k$个专家对准则$i$到准则$j$的评估

## 第二阶段：模糊DEMATEL（准则级）

### 3. 初始直接关系矩阵构建
**易懂描述：** 将聚合后的模糊数分别存储到三个矩阵中
```
X_l矩阵 = 所有下界值组成的矩阵
X_m矩阵 = 所有中值组成的矩阵  
X_u矩阵 = 所有上界值组成的矩阵
```

**LaTeX表达式：**
```latex
\tilde{X} = (X^l, X^m, X^u)
```
其中 $X^l_{ij} = l_{ij}$，$X^m_{ij} = m_{ij}$，$X^u_{ij} = u_{ij}$

### 4. 标准化直接关系矩阵
**易懂描述：** 用每个矩阵的最大行和来标准化，确保收敛
```
标准化矩阵 = 原矩阵 / max(每行元素的和)
```

**LaTeX表达式：**
```latex
\tilde{X}^{norm} = \frac{\tilde{X}}{\max_i \sum_{j=1}^{n} \tilde{x}_{ij}}
```

**代码对应：**
```python
# lines 38-40
X_a = X_a / np.max(np.sum(X_a, axis=1))
X_b = X_b / np.max(np.sum(X_b, axis=1))  
X_c = X_c / np.max(np.sum(X_c, axis=1))
```

### 5. 计算总关系矩阵
**易懂描述：** 计算直接和间接影响的总和
```
Y矩阵 = (单位矩阵 - 标准化X矩阵)的逆矩阵
总关系矩阵T = 标准化X矩阵 × Y矩阵
```

**LaTeX表达式：**
```latex
\tilde{Y} = (I - \tilde{X}^{norm})^{-1}
```
```latex
\tilde{T} = \tilde{X}^{norm} \times \tilde{Y}
```

**等价形式：**
```latex
\tilde{T} = \tilde{X}^{norm} + (\tilde{X}^{norm})^2 + (\tilde{X}^{norm})^3 + \cdots
```

**代码对应：**
```python
# lines 42-47
Y_a = np.linalg.inv(np.identity(len(final_list)) - X_a)
Y_b = np.linalg.inv(np.identity(len(final_list)) - X_b)
Y_c = np.linalg.inv(np.identity(len(final_list)) - X_c)

T_a = np.matmul(X_a, Y_a)
T_b = np.matmul(X_b, Y_b)
T_c = np.matmul(X_c, Y_c)
```

### 6. 计算影响度和被影响度
**易懂描述：** 
```
影响度D = 总关系矩阵每行的和（该准则对其他准则的总影响）
被影响度R = 总关系矩阵每列的和（该准则受其他准则的总影响）
```

**LaTeX表达式：**
```latex
D_i = \sum_{j=1}^{n} t_{ij} \quad \text{(影响度)}
```
```latex
R_j = \sum_{i=1}^{n} t_{ij} \quad \text{(被影响度)}
```

**代码对应：**
```python
# lines 49-55
D_a = np.sum(T_a, axis=1)  # 按行求和
D_b = np.sum(T_b, axis=1)
D_c = np.sum(T_c, axis=1)
R_a = np.sum(T_a, axis=0)  # 按列求和
R_b = np.sum(T_b, axis=0)
R_c = np.sum(T_c, axis=0)
```

### 7. 解模糊化
**易懂描述：** 将三角模糊数转换为清晰值（重心法）
```
清晰值 = 下界值 + (上界值-下界值 + 中值-下界值) / 3
```

**LaTeX表达式：**
```latex
defuzzied = l + \frac{(u-l) + (m-l)}{3}
```

**代码对应：**
```python
# lines 56-57
defuzzied_D = (((D_c - D_a) + (D_b - D_a)) / 3) + D_a
defuzzied_R = (((R_c - R_a) + (R_b - R_a)) / 3) + R_a
```

### 8. 计算重要性和影响关系指标
**易懂描述：**
```
重要性指标 = 影响度 + 被影响度  (D+R，表示准则的重要程度)
影响关系指标 = 影响度 - 被影响度  (D-R，正值为原因，负值为结果)
```

**LaTeX表达式：**
```latex
Prominence_i = D_i + R_i \quad \text{(重要性)}
```
```latex
Relation_i = D_i - R_i \quad \text{(影响关系)}
```

**代码对应：**
```python
# lines 58-59
D_plus_R = defuzzied_D + defuzzied_R
D_minus_R = defuzzied_D - defuzzied_R
```

## 第三阶段：模糊DEMATEL（维度级）

### 9. 构建维度关系矩阵
**易懂描述：** 从准则级总关系矩阵中提取子矩阵，计算维度间关系
```
维度i到维度j的关系 = 对应子矩阵中所有元素的平均值
```

**LaTeX表达式：**
```latex
T^d_{ij} = \frac{1}{n_i \times n_j} \sum_{p \in D_i} \sum_{q \in D_j} T_{pq}
```
其中 $D_i$ 表示维度$i$包含的准则集合，$n_i$ 为维度$i$的准则数量

**代码对应：**
```python
# lines 115-120
Td_a_list = cp.sub_matrices(dimension_count, criteria_count, T_a)
Td_b_list = cp.sub_matrices(dimension_count, criteria_count, T_b)
Td_c_list = cp.sub_matrices(dimension_count, criteria_count, T_c)

Td_a = cp.td_maker(dimension_count, Td_a_list)
Td_b = cp.td_maker(dimension_count, Td_b_list)
Td_c = cp.td_maker(dimension_count, Td_c_list)
```

### 10. 维度级影响度计算
**易懂描述：** 与准则级相同的计算方法，但应用于维度矩阵
```
维度影响度 = 维度关系矩阵每行的和
维度被影响度 = 维度关系矩阵每列的和
```

**LaTeX表达式：**
```latex
D^d_i = \sum_{j=1}^{m} T^d_{ij} \quad \text{(维度影响度)}
```
```latex
R^d_j = \sum_{i=1}^{m} T^d_{ij} \quad \text{(维度被影响度)}
```
其中 $m$ 为维度数量

**代码对应：**
```python
# lines 127-133
Td_D_a = np.sum(Td_a, axis=1)
Td_D_b = np.sum(Td_b, axis=1)
Td_D_c = np.sum(Td_c, axis=1)
Td_R_a = np.sum(Td_a, axis=0)
Td_R_b = np.sum(Td_b, axis=0)
Td_R_c = np.sum(Td_c, axis=0)
```

## 第四阶段：ANP超矩阵计算

### 11. 子矩阵标准化
**易懂描述：** 对每个子矩阵进行行标准化，使每行和为1
```
标准化元素 = 原元素 / 该行所有元素的和
```

**LaTeX表达式：**
```latex
w_{ij}^{norm} = \frac{w_{ij}}{\sum_{k=1}^{n_j} w_{ik}}
```

**代码对应：**
```python
# computation.py lines 56-62
def normalizing(matrix):
    norm_mat = np.zeros((len(matrix), len(matrix[0])))
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            norm_mat[i, j] = (matrix[i, j] / np.sum(matrix[i]))
    return norm_mat
```

### 12. 构建超矩阵
**易懂描述：** 将所有标准化的子矩阵按维度顺序组合成大矩阵
```
超矩阵 = [W11  W12  ...  W1m]
        [W21  W22  ...  W2m]
        [...  ...  ...  ...]
        [Wm1  Wm2  ...  Wmm]
```

**LaTeX表达式：**
```latex
W = \begin{bmatrix}
W_{11} & W_{12} & \cdots & W_{1m} \\
W_{21} & W_{22} & \cdots & W_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
W_{m1} & W_{m2} & \cdots & W_{mm}
\end{bmatrix}
```
其中 $W_{ij}$ 为维度$i$到维度$j$的子矩阵

### 13. 无权重超矩阵
**易懂描述：** 将标准化超矩阵转置，改变影响方向
```
无权重超矩阵 = 标准化超矩阵的转置
```

**LaTeX表达式：**
```latex
W^{unweighted} = (W^{normalized})^T
```

**代码对应：**
```python
# lines 166-168
unweighted_l = normal_super_l.transpose()
unweighted_m = normal_super_m.transpose()
unweighted_u = normal_super_u.transpose()
```

### 14. 维度权重矩阵处理
**易懂描述：** 标准化维度关系矩阵并转置，用作权重
```
维度权重矩阵 = 标准化(维度关系矩阵)的转置
```

**LaTeX表达式：**
```latex
T^{d,norm} = normalize(T^d)
```
```latex
W^{cluster} = (T^{d,norm})^T
```

**代码对应：**
```python
# lines 175-182
Td_l_norm = cp.normalizing(Td_a)
Td_m_norm = cp.normalizing(Td_b)
Td_u_norm = cp.normalizing(Td_c)

Td_l_norm_trans = Td_l_norm.transpose()
Td_m_norm_trans = Td_m_norm.transpose()
Td_u_norm_trans = Td_u_norm.transpose()
```

### 15. 加权超矩阵
**易懂描述：** 用维度权重对无权重超矩阵进行加权
```
加权子矩阵 = 无权重子矩阵 × 对应的维度权重元素
```

**LaTeX表达式：**
```latex
W^{weighted}_{ij} = W^{unweighted}_{ij} \times w^{cluster}_{ij}
```

**代码对应：**
```python
# lines 188-192
weighted_l_list = cp.weigh_super_list(un_sup_l_list, Td_l_norm_trans, dimension_count)
weighted_m_list = cp.weigh_super_list(un_sup_m_list, Td_m_norm_trans, dimension_count)
weighted_u_list = cp.weigh_super_list(un_sup_u_list, Td_u_norm_trans, dimension_count)
```

### 16. 极限超矩阵
**易懂描述：** 将加权超矩阵提升到足够高的幂次直到收敛
```
极限超矩阵 = 加权超矩阵的k次幂 (k足够大)
```

**LaTeX表达式：**
```latex
W^{limit} = \lim_{k \to \infty} (W^{weighted})^k
```

**实际计算：**
```latex
W^{limit} = (W^{weighted})^{101}
```

**代码对应：**
```python
# lines 200-202
limit_weighted_l = np.linalg.matrix_power(weighted_l, 101)
limit_weighted_m = np.linalg.matrix_power(weighted_m, 101)
limit_weighted_u = np.linalg.matrix_power(weighted_u, 101)
```

### 17. 收敛性检查
**易懂描述：** 检查矩阵是否稳定（每行元素差异小于阈值）
```
如果 |矩阵[i,j] - 矩阵[i,j+1]| > 0.0001，则继续迭代
```

**LaTeX表达式：**
```latex
|w^{limit}_{i,j} - w^{limit}_{i,j+1}| \leq \epsilon, \quad \forall i,j
```
其中 $\epsilon = 0.0001$

**代码对应：**
```python
# lines 204-225
while x < len(criteria_names):
    z = 0
    while z < (len(criteria_names) - 1):
        if limit_weighted_l[x, z] - limit_weighted_l[x, z + 1] > 0.0001:
            limit_weighted_l = np.linalg.matrix_power(weighted_l, 101)
            x, z = 0, 0
        # ... 类似检查其他矩阵
        z += 1
    x += 1
```

### 18. 最终权重计算
**易懂描述：** 对极限超矩阵进行解模糊化，提取权重向量
```
最终权重 = 解模糊化(极限超矩阵的第一列)
```

**LaTeX表达式：**
```latex
w^{final}_i = defuzzify(w^{limit,l}_i, w^{limit,m}_i, w^{limit,u}_i)
```
```latex
w^{final}_i = w^{limit,l}_i + \frac{(w^{limit,u}_i - w^{limit,l}_i) + (w^{limit,m}_i - w^{limit,l}_i)}{3}
```

**代码对应：**
```python
# lines 259-261
defuzzy_lim_weighted = ((((limit_weighted_u - limit_weighted_l) + 
                         (limit_weighted_m - limit_weighted_l)) / 3) 
                        + limit_weighted_l)
```

**最终输出：**
```python
# lines 263-265
do.exl_out(defuzzy_lim_weighted[:, 0], 'defuzzy_lim_weighted', criteria_names, header)
```

## 总结

这套DANP算法包含18个主要计算步骤，从模糊化开始，经过DEMATEL分析识别因果关系，再通过ANP计算最终权重。每个步骤都有明确的数学基础和实际意义，但在维度准则数量不平衡时可能产生偏差，需要在应用时特别注意。