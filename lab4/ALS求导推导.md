# ALS 求导过程推导
内容主要是对发表ALS方法的论文[1]做一些整理、摘录。所以符号表示方式上，尽量跟论文一致。


## 0.符号表示

|          符号           |           含义 / 解释          |
|     :-----------:      |         :-------------         |
| <img src="http://latex.codecogs.com/gif.latex?n_{u}"> | 用户数量 <img src="http://latex.codecogs.com/gif.latex?n_{users}"> | 
| <img src="http://latex.codecogs.com/gif.latex?n_{m}"> | 电影数量 <img src="http://latex.codecogs.com/gif.latex?n_{items}"> |
| <img src="http://latex.codecogs.com/gif.latex?K"> | 隐含特征个数 |
| <img src="http://latex.codecogs.com/gif.latex?R"> | 评分矩阵, <img src="http://latex.codecogs.com/gif.latex?R\in\mathbb{R}^{n_{u}\times%20n_m}"> <br/> <img src="http://latex.codecogs.com/gif.latex?R_{i,j}"> 表示用户 <img src="http://latex.codecogs.com/gif.latex?i"> 对电影 <img src="http://latex.codecogs.com/gif.latex?j"> 的评分 |
| <img src="http://latex.codecogs.com/gif.latex?U"> | （用户，特征）矩阵， <img src="http://latex.codecogs.com/gif.latex?U\in\mathbb{R}^{K\times%20n_{u}}"> <br/> <img src="http://latex.codecogs.com/gif.latex?U_{i,k}"> 表示用户 <img src="http://latex.codecogs.com/gif.latex?i">  对隐含特征 <img src="http://latex.codecogs.com/gif.latex?k"> 的喜好程度，<br/>**列向量** <img src="http://latex.codecogs.com/gif.latex?u_{i}"> 中包含的所有隐含特征刻画了用户i在推荐系统中的表示形式 |
| <img src="http://latex.codecogs.com/gif.latex?M"> | （特征，电影）矩阵， <img src="http://latex.codecogs.com/gif.latex?M\in\mathbb{R}^{K\times%20n_m}"> <br/> <img src="http://latex.codecogs.com/gif.latex?M_{k,j}"> 表示电影 <img src="http://latex.codecogs.com/gif.latex?j"> 符合隐含特征 <img src="http://latex.codecogs.com/gif.latex?k"> 的程度，<br/>**列向量** <img src="http://latex.codecogs.com/gif.latex?m_{j}"> 中包含的所有隐含特征刻画了电影j在推荐系统中的表示形式 |
| <img src="http://latex.codecogs.com/gif.latex?I"> | 原始评分矩阵中有评分存在的 <img src="http://latex.codecogs.com/gif.latex?\left(i,j\right)"> 对，即（用户， 电影）元组的集合 |
| <img src="http://latex.codecogs.com/gif.latex?I_i^U"> | 用户 <img src="http://latex.codecogs.com/gif.latex?i"> 评价过的**所有电影**的下标组成的集合 |
| <img src="http://latex.codecogs.com/gif.latex?I_j^M"> | 评价过电影 <img src="http://latex.codecogs.com/gif.latex?j"> 的**所有用户**下标组成的集合 |
| <img src="http://latex.codecogs.com/gif.latex?M_{I_i^U}" > | 从电影特征矩阵 <img src="http://latex.codecogs.com/gif.latex?M"> 中抽取用户 <img src="http://latex.codecogs.com/gif.latex?i"> 评价过的所有电影的特征向量，组成一个小型矩阵 <img src="http://latex.codecogs.com/gif.latex?M_{I_i^U}" > |
| <img src="http://latex.codecogs.com/gif.latex?U_{I_i^M}"> | 从用户特征矩阵 <img src="http://latex.codecogs.com/gif.latex?U"> 中抽取所有评价过电影 <img src="http://latex.codecogs.com/gif.latex?j"> 的用户的特征向量，组成一个小型矩阵 <img src="http://latex.codecogs.com/gif.latex?U_{I_i^M}" > |                             
| <img src="http://latex.codecogs.com/gif.latex?R^T\left(i,I_i^U\right)"> | 用户 <img src="http://latex.codecogs.com/gif.latex?i"> **真实评价过**的分数组成的行向量，是行向量 <img src="http://latex.codecogs.com/gif.latex?R_i"> 的一部分 |
| <img src="http://latex.codecogs.com/gif.latex?R\left(I_i^M,j\right)"> | 电影 <img src="http://latex.codecogs.com/gif.latex?j"> 的所有真实评分组成的列向量，是列向量 <img src="http://latex.codecogs.com/gif.latex?\mathbf{r}_j"> 的一部分 |  
| <img src="http://latex.codecogs.com/gif.latex?E"> | 单位矩阵，<img src="http://latex.codecogs.com/gif.latex?E\in%20\mathbb{R}^{K\times%20K}">   |

## 1.推荐系统矩阵分解

在推荐系统的矩阵分解方法中，原始评分矩阵R被分解成了U和M两个矩阵。

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?R=U^T\times%20M">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?R\in\mathbb{R}^{n_{u}\times%20n_m}">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?U\in\mathbb{R}^{K\times%20n_{u}}">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?M\in\mathbb{R}^{K\times%20n_m}">
</div>


<!--
### 模型的评估和优化
-->

### 1.1. 损失函数（loss function）

采用平方误差（square loss）函数来评估推荐结果对单个评分的损失值

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?e_{i,j}=\left(R_{i,j}-\hat{R_{i,j}}\right)^2=\left(R_{i,j}-\sum_{k=1}^{K}{U_{k,i}M_{k,j}}\right)^2">
</div>

>其中 <img src="http://latex.codecogs.com/gif.latex?\mathbf{u}_i,\mathbf{m}_k"> 表示的都是列向量，而 <img src="http://latex.codecogs.com/gif.latex?\mathbf{u}_i^T">
表示列向量的转置。

对损失函数作正则化处理，有：

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\begin{align*}e_{i,j}&=\left(R_{i,j}-\sum_{k=1}^{K}{U_{k,i}M_{k,j}}\right)^2+\lambda\sum_{k=1}^{K}{\left(U_{k,i}^2+M_{k,j}^2\right)}\\&=\left(R_{i,j}-\mathbf{u}_i^T\mathbf{m}_j\right)^2+\lambda\left(\left|\mathbf{u}_i\right|^2+\left|\mathbf{m}_{k}\right|^2\right)\end{align*}">
</div>

### 1.2. 目标函数（objective function）

累加所有原始评分矩阵中**出现过的分数**的误差，得到目标函数：

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\begin{align*}f\left(U,M\right)&=\sum_{\left(i,j\right)\in%20I}e_{i,j}\\&=\sum_{\left(i,j\right)\in%20I}\left(R_{i,j}-\mathbf{u}_i^T\mathbf{m}_j\right)^2+\lambda\left(\sum_{i}n_{u_i}\left|\mathbf{u}_i\right|^2+\sum_{j}n_{m_j}\left|\mathbf{m}_{k}\right|^2\right)\end{align*}">
</div>

>注解：由于在 <img src="http://latex.codecogs.com/gif.latex?e_{i,j}"> 中，<img src="http://latex.codecogs.com/gif.latex?\left|\mathbf{u}_i\right|^2"> 贡献了一次，那么在行向量 <img src="http://latex.codecogs.com/gif.latex?\mathbf{r}_{i}"> 的损失 <img src="http://latex.codecogs.com/gif.latex?e_{i}^T"> 中，<img src="http://latex.codecogs.com/gif.latex?\left|\mathbf{u}_i\right|^2"> 会贡献 <img src="http://latex.codecogs.com/gif.latex?n_{u_i}"> 次。同理，在列向量 <img src="http://latex.codecogs.com/gif.latex?\mathbf{r}_{j}"> 的损失 <img src="http://latex.codecogs.com/gif.latex?e_{j}"> 中，<img src="http://latex.codecogs.com/gif.latex?\left|\mathbf{m}_j\right|^2"> 会贡献 <img src="http://latex.codecogs.com/gif.latex?n_{m_j}"> 次。

### 1.3. 成本函数（cost function）

模型的成本函数（经验风险）可以用RMSE（root-mean-square error 均方根差）表示为：

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\begin{align*}RMSE\left(U,M\right)&=\sqrt{\frac{\sum_{\left(i,j\right)\in%20I}e_{i,j}}{\left|I\right|}}=\sqrt{\frac{\sum_{\left(i,j\right)\in%20I}\left(R_{i,j}-\mathbf{u}_i^T\mathbf{m}_j\right)^2}{\left|I\right|}}\end{align*}">
</div>

>* 该方法用于评估训练集时，误差被称为RMSD（root-mean-square deviation 均方根偏差）
>* 该方法用于评估测试集时，误差被称为RMSE（root-mean-square error 均方根误差）

即对原始评分矩阵中（除去空值）所有**出现过的评分**和预测评分的【均方误差的值】开根号。该值越接近0，模型的效果越好。发表ALS方法的论文[1]称，用1000个隐含特征作矩阵分解，并用ALS的方法训练，在netflix prize dataset上得到的RMSE为0.8985。

>评分的范围为1分到5分，假如用模型预测出来的每个预测值和**真实值**都有2分的误差，那么该模型的RMSE为2
>
<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?RMSE\left(U,M\right)=\sqrt{\frac{1}{\left|I\right|}\sum_{\left(i,j\right)\in%20I}2^2}=\sqrt{\frac{4}{\left|I\right|}\sum_{\left(i,j\right)\in%20I}1}=2">
</div>


## 2.ALS求导
ALS (alternating-least-squares 交替最小平方)的训练步骤是： 
1.  用较小的随机数初始化矩阵U和M。用每一部电影的平均评分（注意排除空值）作为该电影的第一个隐含特征。
2. 固定M，用**目标函数**对矩阵U的偏导数，优化矩阵U
3. 固定U，用**目标函数**对矩阵M的偏导数，优化矩阵M
4. 重复step2, step3，直到满足停止条件（例如训练集上的RMSD足够小）

以下着重关注求导部分

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?f\left(U,M\right)=\sum_{\left(i,j\right)\in%20I}\left(R_{i,j}-\mathbf{u}_i^T\mathbf{m}_j\right)^2+\lambda\left(\sum_{i}n_{u_i}\left|\mathbf{u}_i\right|^2+\sum_{j}n_{m_j}\left|\mathbf{m}_{k}\right|^2\right)">
</div>

<img src="img/ui_derivation.png">

以上式子出现的<img src="http://latex.codecogs.com/gif.latex?M_{I_i^U}, R^T\left(i,I_i^U\right), E"> 等变量的说明，见 [0.符号表示](#0符号表示)

[变换1]的说明
<img src="img/eq1.png">

<!--
<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?MM^Tu_i=M\hat{R}_i=\begin{bmatrix}M_1\hat{R}_i\\M_2\hat{R}_i\\...\\M_k\hat{R}_i\\...\\M_K\hat{R}_i\end{bmatrix}\quad\Rightarrow\quad\begin{bmatrix}\sum_{j\in%20I_i^U}m_{1,j}\hat{R}_i\\\sum_{j\in%20I_i^U}m_{2,j}\hat{R}_i\\...\\\sum_{j\in%20I_i^U}m_{k,j}\hat{R}_i\\...\\\sum_{j\in%20I_i^U}m_{K,j}\hat{R}_i\end{bmatrix}=\begin{bmatrix}\sum_{j\in%20I_i^U}m_{1,j}m_j^Tu_i\\\sum_{j\in%20I_i^U}m_{2,j}m_j^Tu_i\\...\\\sum_{j\in%20I_i^U}m_{k,j}m_j^Tu_i\\...\\\sum_{j\in%20I_i^U}m_{K,j}m_j^Tu_i\end{bmatrix}">
</div>
-->

### 2.1. 列向量ui的更新

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathbf{u}_i=A_i^{-1}V_i">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?A_i=M_{I_i^U}M_{I_i^U}^T+\lambda%20n_{u_i}E">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?V_i=M_{I_i^U}R^T\left(i,I_i^U\right)">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?E\in%20\mathbb{R}^{K\times%20K}">
</div>


### 2.2. 列向量mj的更新

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathbf{m}_j=A_j^{-1}V_j">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?A_j=U_{I_i^M}U_{I_i^M}^T+\lambda%20n_{m_j}E">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?V_j=U_{I_i^M}R^T\left(I_i^M,j\right)">
</div>

<div class="eq" align="center">
    <img src="http://latex.codecogs.com/gif.latex?E\in%20\mathbb{R}^{K\times%20K}">
</div>

## 3.Python代码实现（慎点）



## 4.实验结果



## 5.Netflix Prize 简述

>The Netflix Prize is a large-scale data mining competition held by Netflix
for the best recommendation system algorithm for predicting user ratings on
movies, based on a training set of more than 100 million ratings given by over
480,000 users to 17,700 movies. Each training data point consists of a quadruple
(user, movie, date, rating) where rating is an integer from 1 to 5. The test
dataset consists of 2.8 million data points with the ratings hidden. The goal is
to minimize the RMSE (root mean squared error) when predicting the ratings
on the test dataset. Netflix's own recommendation system (CineMatch) scores
0.9514 on the test dataset, and the grand challenge is to improve it by 10%.

摘自论文[1]中的描述，大致状况就是：在2006年的时候，Netflix公司发布了个百万美元大赛，只要能够做到比它们原来的推荐系统的性能更优10%（用RMSE衡量是0.85626），即可拿奖。2008年，利用ALS的方法，在Linux服务器集群上利用GPU矩阵运算，该论文的作者们取得了5.91%的提升。最终这个提升10%的难题在2009年才被解决[2]。

## 6.Reference
1. [Large-Scale Parallel Collaborative Filtering for the Netflix Prize](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Large-Scale+Parallel+Collaborative+Filtering+for+the+Netflix+Prize&btnG=) 
2. [Netflix Prize](https://www.netflixprize.com/)
3. [NetFlix百万美金数据建模大奖的故事](https://mp.weixin.qq.com/s?src=3&timestamp=1543407206&ver=1&signature=ZFCFiBH6wqYd0X*s6hU3mhvmyiVrrVpOK5sbGgAku7JjMq0430qfHiDGUdacIO8bYlHLakerpzZMUPNMUIjyW2I06v6V359eUCIldOySPvOBELOwEygw9b1ZEmZDRVWJE8sqDTYYNmV1aWmwy0UZVw==)


