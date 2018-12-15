# lab4

- [2.Methods and Theory](#2methods-and-theory)
  - [2.1. Matrix Factorization Technique](#21-matrix-factorization-technique)
  - [2.2. Loss Function and Objective Function](#22-loss-function-and-objective-function)
  - [2.3. Optimization Methods](#23-optimization-methods)
    - [2.3.1. SGD for Matrix Factorization](#231-sgd-for-matrix-factorization)
    - [2.3.2. ALS for Matrix Factorization](#232-als-for-matrix-factorization)
- [3.Experiment](#3experiment)
  - [3.1. Dataset](#31-dataset)
  - [3.2. Experiment Step](#32-experiment-step)
    - [3.2.1. Training procedure](#321-training-procedure)
    - [3.2.2. Core Code of SGD optimization method** (written in python)](#322-core-code-of-sgd-optimization-method-written-in-python)
    - [3.2.3. Core Code of ALS optimization method** (written in python)](#323-core-code-of-als-optimization-method-written-in-python)
  - [3.3. Experiment Results](#33-experiment-results)
    - [3.3.1. Result of SGD method](#331-result-of-sgd-method)
    - [3.3.2. Result of ALS method](#332-result-of-als-method)
    - [3.3.3. Comparison between SGD and ALS methods](#333-comparison-between-sgd-and-als-methods)

## 2.Methods and Theory

### 2.1. Matrix Factorization Technique
Matrix factorization is a class of collaborative filtering algorithms used in recommender systems.[1]. 

Given a rating Matrix $R\in \mathbb{R}^{m\times n}$, with sparse ratings from $m$ users to $n$ items, matrix factorization algorithms work by decomposing $R$ into the product of two low-rank feature matrices $P\in \mathbb{R}^{m\times k}$ and $Q\in\mathbb{R}^{k\times n}$, where $k$ represents the predefined number of latent features.

$$R=PQ$$

Matrix factorization is a power tool that allows us to discover the latent features underlying the interactions between users and items.

### 2.2. Loss Function and Objective Function
Here we use **square error loss** to evaluate the recommendation model.
$$e_{u,i}=\left(r_{u,i}-\hat{r}_{u,i}\right)^2=\left(r_{u,i}-\mathbb{p}^T_u\mathbb{q}_i\right)^2$$
$r_{u,i}$ denotes the **actual rating** of user $u$ for item $i$ and $\hat{r}_{u,i}$ denotes the prediction rating. $\mathbb{p}^T_u$ is a row vector of matrix $P$, indicating the latent features of user $u$. Similarily, $\mathbb{q}_i$ represents a column vector of matrix $Q$ indicating the latent features of item $i$.

Then the **root mean square error (RMSE)** is selected as the empirical risk function. In the following equation, $E$ is the indice set containing $(u, i)$ pairs which indicate user $u$ has rated item $i$ in the rating matrix.

$$RMSE\left(R,P,Q\right)=\sqrt{\frac{1}{\left|E\right|}\sum_{\left(u,i\right)\in E}e_{u,i}}=\sqrt{\frac{\sum_{\left(u,i\right)\in E}\left(r_{u,i}-\mathbf{p}_u^T\mathbf{q}_i\right)^2}{\left|E\right|}}$$

Our goal is to minimize the following objective function:
$$L=\sum_{u,i}{\left(r_{u,i}-\mathbb{p}^T_u\mathbb{q}_i\right)^2}+\lambda\left(\sum_u{n_{\mathbb{p}_u}\left|\mathbb{p}_u\right|^2}+\sum_i{n_{\mathbb{q}_i}\left|\mathbb{q}_i\right|^2}\right) \quad \left(1\right)$$

$\lambda$ is regularization parameter to avoid overtting, while
$n_{\mathbb{p}_u}$ and $n_{\mathbb{q}_i}$ denote the number of total ratings on user $u$ and item $i$, respectively.

### 2.3. Optimization Methods
**Stochastic gradient descent (SGD)** and **alternating least squares (ALS)** are two methods commonly used to solve the matrix factorization problem. In this experiment, we will exhibit ideas and methodolies of both methods, and then use them to perform experiments.

#### 2.3.1. SGD for Matrix Factorization
The general steps of SGD are listed as follow:
1. Initialize matrix P and matrix Q with random numbers for each entry, set up parameters $\lambda_p$, $\lambda_q$, learning rate $\eta$ and maximum number of iterations.
2. Randomly select an observed sample $r_{u,i}$ from observed set.
3. Calculate the gradient to the objective function: 
$$\frac{\partial L}{\partial\mathbf{p}_u}=e_{u,i}\left(-\mathbf{q}_i\right)+\lambda_p\mathbf{p}_u$$ 
$$\frac{\partial L}{\partial\mathbf{q}_i}=e_{u,i}\left(-\mathbf{p}_u\right)+\lambda_q\mathbf{q}_i$$
4. Update the feature matrices P and Q with learning rate $\eta$: 
$$\mathbf{p}_u=\mathbf{p}_u+\eta(e_{u,i}\mathbf{q}_i-\lambda_p\mathbf{p}_u)$$
$$\mathbf{q}_i=\mathbf{q}_i+\eta(e_{u,i}\mathbf{p}_u-\lambda_q\mathbf{q}_i)$$
5. Repeat the step 2 to step 4 until convergence.

In this experiment, we set $\lambda_p$ and $\lambda_q$ equal so the objective function is the same as equation [1]. 

#### 2.3.2. ALS for Matrix Factorization
The procedure of ALS is described as follow:
1. Initialize matrix P with small random numbers for each entry. Then initialize matrix Q by assigning the average rating for that movie as the first row, and small random numbers for the remaining entries.
2. Optimize P while fixing Q. Update each row vector $p_u^T$ of $P$ with respect to the following equation:
$$p_u^T=\left(\left(Q_{u*}Q_{u*}^T+\lambda n_{p_u}I\right)^{-1}Q_{u*}R_{u*}^T\right)^T$$

>$Q_{u*}\in \mathbb{R}^{k\times n_{p_u}}$ denotes the sub-matrix of $Q$ where columns $i$, $\left(u, i\right)\in E$ are selected, and $R_{u*}\in \mathbb{R}^{n_{p_u}}$ is the row vector where entry $i$, $\left(u, i\right)\in E$ of the $u$-th row of $R$, $R_u^T$, are selected.

3. Similarily, optimize Q while fixing P. Update each column vector $q_i$ of $Q$ with respect to the following equation:
$$q_i=\left(P_{i*}^TP_{i*}+\lambda n_{q_i}I\right)^{-1}P_{i*}^TR_{i*}^T$$

>$P_{i*}\in \mathbb{R}^{n_{q_i}\times k}$ denotes the sub-matrix of $P$ where each row $u$, $\left(u, i\right)\in E$ are selected, and $R_{i*}\in \mathbb{R}^{n_{q_i}}$ is the column vector where each entry $u$, $\left(u, i\right)\in E$ of the $i$-th column of $R$, $R_i$, are selected.

4. Repeat Steps 2 and 3 until convergence or a stopping criterion is satisfied.

For more details about the process of derivation, check the paper[2].
 
## 3.Experiment

### 3.1. Dataset
The dataset used in this experiment is from [MovieLens-100k](http://files.grouplens.org/datasets/movielens) dataset. Its sub-dataset, u.data, consists of 10,000 ratings from 943 users out of 1682 movies. At least, each user has rated 20 movies.

### 3.2. Experiment Step

#### 3.2.1. Training procedure
1. Read the training set u1.base and testing set u1.test. Populate the original scoring matrix $R_{train}\in \mathbb{R}^{m\times n}$ and $R_{test}\in \mathbb{R}^{m\times n}$ against the raw data, and fill 0 for null values.
2. Initialize the user factor matrix $P\in \mathbb{R}^{m\times k}$ and the item (movie) factor matrix $Q\in \mathbb{R}^{k\times n}$, where $k$ is the number of potential features.
3. Determine the hyperparameters learning rate $\eta$ and the penalty factor $\lambda$.
4. Use alternate least squares or stochastic gradient descent optimization method to decompose the sparse user score matrix, get the user factor matrix and item (movie) factor matrix. 
5. Repeat step 4 several times, get a satisfactory user factor matrix $P$ and an item factor matrix $Q$. Draw the loss curve with respect to both training and testing procedures during varying iterations.
6. The final score prediction matrix $\hat{R}$ is obtained by multiplying the user factor matrix $P$ and the item factor matrix $Q$.

#### 3.2.2. Core Code of SGD optimization method** (written in python)

**TODO:**

```python
```

#### 3.2.3. Core Code of ALS optimization method** (written in python)

**TODO:**

```python
```

### 3.3. Experiment Results

#### 3.3.1. Result of SGD method
**TODO:**
 
#### 3.3.2. Result of ALS method
**TODO:**

#### 3.3.3. Comparison between SGD and ALS methods
**TODO:**

# References
1. [Wikipedia. Matrix_factorization.](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
2. Yunhong Zhou et al. Large-Scale Parallel Collaborative Filtering forthe Netflix Prize. 
