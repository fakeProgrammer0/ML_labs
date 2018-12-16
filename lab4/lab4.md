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
    - [3.2.2. Core Code of SGD optimization method (written in python)](#322-core-code-of-sgd-optimization-method-written-in-python)
    - [3.2.3. Core Code of ALS optimization method (written in python)](#323-core-code-of-als-optimization-method-written-in-python)
  - [3.3. Experiment Results](#33-experiment-results)
    - [3.3.1. Result of SGD method](#331-result-of-sgd-method)
    - [3.3.2. Result of ALS method](#332-result-of-als-method)
    - [3.3.3. Tuning Parameters of ALS](#333-tuning-parameters-of-als)
      - [3.3.3.1. Penlaty Facotr $\lambda$](#3331-penlaty-facotr-lambda)
      - [3.3.3.2. The number of Latent Features $K$](#3332-the-number-of-latent-features-k)
    - [3.3.4. Comparison between SGD and ALS methods](#334-comparison-between-sgd-and-als-methods)
- [References](#references)
  - [1. Wikipedia. Matrix_factorization.)](#1-wikipedia-matrix_factorization)
  - [2. Yunhong Zhou et al. Large-Scale Parallel Collaborative Filtering for the Netflix Prize.](#2-yunhong-zhou-et-al-large-scale-parallel-collaborative-filtering-for-the-netflix-prize)

## 2.Methods and Theory

### 2.1. Matrix Factorization Technique
Matrix factorization is a class of collaborative filtering algorithms used in recommender systems.[\[1\]](#1-wikipedia-matrix_factorization). 

Given a rating Matrix $R\in \mathbb{R}^{m\times n}$, with sparse ratings from $m$ users to $n$ items, matrix factorization algorithms work by decomposing $R$ into the product of two low-rank feature matrices $P\in \mathbb{R}^{m\times k}$ and $Q\in\mathbb{R}^{k\times n}$, where $k$ represents the predefined number of latent features.

$$R=PQ$$

Matrix factorization is a power tool that allows us to discover the latent features underlying the interactions between users and items.

### 2.2. Loss Function and Objective Function
Here we use **square error loss** to evaluate the recommendation model.
$$e_{u,i}=\left(r_{u,i}-\hat{r}_{u,i}\right)^2=\left(r_{u,i}-\mathbb{p}^T_u\mathbb{q}_i\right)^2$$
$r_{u,i}$ denotes the **actual rating** of user $u$ for item $i$ and $\hat{r}_{u,i}$ denotes the prediction rating. $\mathbb{p}^T_u$ is a row vector of matrix $P$, indicating the latent features of user $u$. Similarily, $\mathbb{q}_i$ represents a column vector of matrix $Q$ indicating the latent features of item $i$.

Then the **root mean square error (RMSE)** is selected as the empirical risk function. In the following equation, $E$ is the indice set containing $(u, i)$ pairs which indicate user $u$ has rated item $i$ in the rating matrix.

$$RMSE\left(R,P,Q\right)=\sqrt{\frac{1}{\left|E\right|}\sum_{\left(u,i\right)\in E}e_{u,i}^2}=\sqrt{\frac{\sum_{\left(u,i\right)\in E}\left(r_{u,i}-\mathbf{p}_u^T\mathbf{q}_i\right)^2}{\left|E\right|}}$$

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

For more details about the process of derivation, check the paper [\[2\]](#2-yunhong-zhou-et-al-large-scale-parallel-collaborative-filtering-for-the-netflix-prize).
 
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

#### 3.2.2. Core Code of SGD optimization method (written in python)

```python
import numpy as np
import math
import random

def MF_SGD_fit(R_train,
               R_test,
               K,
               learning_rate,
               max_epoch,
               reg_lambda_p,
               reg_lambda_q,
               min_loss_threshold=0.1,
               loss_estimate=MF_RMSE,
               epoch_cnt_per_loss_estimate=1000):
    """
    Fit a rating matrix and optimize the matrix factorization model using SGD method.

    Parameters
    ----------
    R_train : ndarray 
        The groundtruth rating matrix used for training, in shape (n_users, n_items).
    R_test : ndarray 
        The groundtruth rating matrix for testing, in shape (n_users, n_items).
    K : int
        The number of latent features.
    learning_rate : float
        The hyper-parameter to control the velocity of gradient descent process, 
        also called step_size.
    max_epoch : int
        The number of training epoches.
    reg_lambda_p, reg_lambda_q : float
        The regularization parameters of the model cost term.
    min_cost_threshold : float
        When the training cost reaches or is lower than the thresold, training will stop.
    loss_estimate :  callable
        A custom loss evaluation function with following signature (R, P, Q) 
        returns the loss of the matrix factorization model. 
        The default setting is using RMSE.
    epoch_cnt_per_loss_estimate : int
        Loss will be estimated at every epoch count.

    Returns
    -------
    R_pred : ndarray
        The predicted rating matrix.

    losses_dict : dict
        A dict containing the model's losses on training and testing dataset 
        during the training procedure.

    """

    n_users, n_items = R_train.shape

    P, Q = np.random.rand(K, n_users), np.random.rand(K, n_items)

    losses_train = []
    losses_test = []

    # acquire observed rating (u, i) pairs to support random selecting efficently
    observed_rating_ui_pairs = []
    for u in range(n_users):
        for i in range(n_items):
            if R_train[u, i]:
                observed_rating_ui_pairs.append((u, i))

    random.shuffle(observed_rating_ui_pairs)

    for epoch in range(max_epoch):

        u, i = random.choice(observed_rating_ui_pairs)

        e_ui = R_train[u, i] - P[:, u] @ Q[:, i]
        # P[:, u] += learning_rate * (2*e_ui*Q[:, i] - reg_lambda_p*P[:, u])
        # Q[:, i] += learning_rate * (2*e_ui*P[:, u] - reg_lambda_q*Q[:, i])

        P[:, u], Q[:, i] = P[:, u] + learning_rate * (2*e_ui*Q[:, i] - reg_lambda_p*P[:, u]), \
                            Q[:, i] + learning_rate * (2*e_ui*P[:, u] - reg_lambda_q*Q[:, i])

        if epoch % epoch_cnt_per_loss_estimate == 0:
            curr_train_loss = loss_estimate(R_train, P, Q)
            losses_train.append(curr_train_loss)

            curr_val_loss = loss_estimate(R_test, P, Q)
            losses_test.append(curr_val_loss)

        if curr_train_loss < min_loss_threshold:
            break

    R_pred = P.T @ Q

    losses_dict = {
        'losses_train': losses_train,
        'losses_test': losses_test,
    }

    return R_pred, losses_dict

```

#### 3.2.3. Core Code of ALS optimization method (written in python)

```python
def MF_ALS_fit(R_train,
               R_test,
               K,
               reg_lambda,
               max_epoch,
               min_RMSE_threshold=0.1,
               loss_estimate=MF_RMSE):
    """
    Fit a rating matrix and optimize the matrix factorization model using ALS method.

    Parameters
    ----------
    R_train : ndarray 
        The groundtruth rating matrix used for training, in shape (n_users, n_items).
    R_test : ndarray 
        The groundtruth rating matrix for testing, in shape (n_users, n_items).
    K : int
        The number of latent features.
    reg_lambda : float
        The regularization parameter lambda of the model cost term.
    max_epoch : int
        The number of training epoches.
    min_cost_threshold : float
        When the training cost reaches or is lower than the thresold, training will stop.
    loss_estimate :  callable
        A custom loss evaluation function with following signature (R, P, Q) 
        returns the loss of the matrix factorization model. 
        The default setting is using RMSE.

    Returns
    -------
    R_pred : ndarray
        The predicted rating matrix.

    losses_dict : dict
        A dict containing the model's losses on training and testing dataset 
        during the training procedure.

    """
    assert isinstance(R_train, np.ndarray)
    n_users, n_items = R_train.shape

    P, Q = np.random.random((K, n_users)), np.random.random((K, n_items))

    N_U = np.sum(R_train != 0, axis=1)  # the number of specific user's ratings, shape: (n_users)

    N_M = np.sum(R_train != 0, axis=0)  # the number of specific movie's ratings, shape: (n_items)
    
    # set the first column of Q as the average rating of each movie
    Q[0, :] = np.sum(R_train, axis=0) / N_M
    for i in range(n_items):
        # if M[0, i] == np.nan:
        if not (0 <= Q[0, i] <= 5):
            Q[0, i] = 0

    losses_train, losses_test = [], []

    for epoch in range(max_epoch):


        for i in range(n_users):
            M_Ui = None  
            R_Ui = None  
            for j in range(n_items):
                if R_train[i, j]:
                    if M_Ui is not None:
                        M_Ui = np.hstack((M_Ui, Q[:, j:j + 1]))
                        R_Ui = np.hstack((R_Ui, R_train[i:i + 1, j:j + 1]))
                    else:
                        M_Ui = Q[:, j:j + 1]
                        R_Ui = R_train[i:i + 1, j:j + 1]

            if M_Ui is None:
                continue

            Ai = M_Ui.dot(M_Ui.T) + reg_lambda * N_U[i] * np.eye(K)
            Vi = M_Ui.dot(R_Ui.T)

            P[:, i:i + 1] = np.dot(np.matrix(Ai).I.getA(), Vi)

        for j in range(n_items):
            U_Mj = None  
            R_Mj = None  
            for i in range(n_users):
                if R_train[i, j]:
                    if U_Mj is not None:
                        U_Mj = np.hstack((U_Mj, P[:, i:i + 1]))
                        R_Mj = np.vstack((R_Mj, R_train[i:i + 1, j:j + 1]))
                    else:
                        U_Mj = P[:, i:i + 1]
                        R_Mj = R_train[i:i + 1, j:j + 1]

            if U_Mj is None:
                continue

            Aj = np.dot(U_Mj, U_Mj.T) + reg_lambda * N_M[j] * np.eye(K)
            Vj = U_Mj.dot(R_Mj)

            Q[:, j:j + 1] = np.dot(np.matrix(Aj).I.getA(), Vj)

        curr_loss_train = loss_estimate(R_train, P, Q)
        curr_loss_test = loss_estimate(R_test, P, Q)
        losses_train.append(curr_loss_train)
        losses_test.append(curr_loss_test)

        if curr_loss_train <= min_RMSE_threshold:
            break

    R_pred = np.dot(P.T, Q)

    losses_dict = {
        'losses_train': losses_train,
        'losses_test': losses_test,
    }

    return R_pred, losses_dict

```

### 3.3. Experiment Results

#### 3.3.1. Result of SGD method
**TODO:**
 
#### 3.3.2. Result of ALS method
**TODO:**

#### 3.3.3. Tuning Parameters of ALS

##### 3.3.3.1. Penlaty Facotr $\lambda$
Fixing $K$, estimate influences of different penlaty facotrs $\lambda$.

##### 3.3.3.2. The number of Latent Features $K$
Estimate the effect of different $K$ on the matrix factorization model.


#### 3.3.4. Comparison between SGD and ALS methods
**TODO:**

## References
### 1. [Wikipedia. Matrix_factorization.](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
### 2. Yunhong Zhou et al. Large-Scale Parallel Collaborative Filtering for the Netflix Prize. 
