# ALS 求导过程推导

在推荐系统的矩阵分解方法中，原始评分矩阵R被分解成了U和M两个矩阵。

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?R=UM">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?R\in\mathbb{R}^{n_{u}\times%20n_m}">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?U\in\mathbb{R}^{n_{u}\times%20K}">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?M\in\mathbb{R}^{K\times%20n_m}">
</div>

|          符号           |           含义 / 解释          |
|     :-----------:      |                 -------------: |
| <img src="http://latex.codecogs.com/gif.latex?R"> | 评分矩阵，<img src="http://latex.codecogs.com/gif.latex?R_{i,j}"> 表示用户i对电影j的评分 |
| <img src="http://latex.codecogs.com/gif.latex?U"> | （用户，特征）矩阵， <img src="http://latex.codecogs.com/gif.latex?U_{i,k}"> 表示用户i对隐含特征k的喜好程度，<br/>行向量 <img src="http://latex.codecogs.com/gif.latex?u_{i}^T"> 中包含的所有隐含特征刻画了用户i在推荐系统中的表示形式 |
| <img src="http://latex.codecogs.com/gif.latex?M"> | （特征，电影）矩阵， <img src="http://latex.codecogs.com/gif.latex?M_{k,j}"> 表示电影j符合隐含特征k的程度，<br/>列向量 <img src="http://latex.codecogs.com/gif.latex?m_{j}"> 中包含的所有隐含特征刻画了电影j在推荐系统中的表示形式 |
| <img src="http://latex.codecogs.com/gif.latex?n_{u}"> | 用户数量 n_users | 
| <img src="http://latex.codecogs.com/gif.latex?n_{m}"> | 电影数量 n_items |
| <img src="http://latex.codecogs.com/gif.latex?K"> | 隐含特征个数 |
| <img src="http://latex.codecogs.com/gif.latex?I"> | 原始评分矩阵中有评分存在的 <img src="http://latex.codecogs.com/gif.latex?\left(i, j\right)"> 对，即（用户， 电影）元组的集合 |
| <img src="http://latex.codecogs.com/gif.latex?I_i^U"> | 用户i评价过的所有电影的下标组成的集合 |
| <img src="http://latex.codecogs.com/gif.latex?I_j^M"> | 用户i评价过的所有电影的下标组成的集合 |
|                           |                                   |
|                           |                                   |
|                           |                                   |

采用平方误差（square loss）函数来评估推荐结果对单个评分的损失值

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?e_{i,j}=\left(R_{i,j}-\hat{R_{i,j}}\right)^2=\left(R_{i,j}-\sum_{k=1}^{K}{U_{i,k}M_{k,j}}\right)^2">
</div>

对损失函数作正则化处理，有

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\begin{align*}e_{i,j}&=\left(R_{i,j}-\sum_{k=1}^{K}{U_{i,k}M_{k,j}}\right)^2+\lambda\sum_{k=1}^{K}{\left(U_{i,k}^2+M_{k,j}^2\right)}\\&=\left(R_{i,j}-\mathbf{u}_i^T\mathbf{m}_j\right)^2+\lambda\left(\left|\mathbf{u}_i\right|^2+\left|\mathbf{m}_{k}\right|^2\right)\end{align*}">
</div>

累加所有原始评分矩阵中**出现过的分数**的误差，得到目标函数

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\begin{align*}f\left(U,M\right)&=\sum_{\left(i,j\right)\in%20I}e_{i,j}\\&=\sum_{\left(i,j\right)\in%20I}\left(R_{i,j}-\mathbf{u}_i^T\mathbf{m}_j\right)^2+\lambda\left(\sum_{i}n_{u_i}\left|\mathbf{u}_i\right|^2+\sum_{j}n_{m_j}\left|\mathbf{m}_{k}\right|^2\right)\end{align*}">
</div>











