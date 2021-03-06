# Linear Regression and Gradient Descent

## Abstract
In this report, we will solve linear regression using both the closed-form solution and gradient descent method based on a small dataset.
After that, we will further learn to tune some parameters such as the learing rate to optimizate our gradient descent model.  

## 1.Introduction
In statistics, linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression.[2]
Closed-form solution and gradient descent are two methods for solving simple linear regression.
Motivations of the report are listed below:
* Further understand of linear regression ，closed-form solution and Stochastic gradient descent.
* Conduct some experiments under small scale dataset.
* Realize the process of optimization and adjusting parameters.

## 2.Methods and Theory
The equation of simple linear regression can be described as:

<div align="center">
    <img src = "http://latex.codecogs.com/gif.latex?y%20=%20w^{T}X%20+%20b%20\quad%20\eqno{(1)}"/>
</div>

Let <img src = "http://latex.codecogs.com/gif.latex?W%20=%20\left(b;\omega%20\right)" />
and then equation (1) can be changed into 
<div align="center">
    <img src = "http://latex.codecogs.com/gif.latex?y%20=%20W%20^T%20X%20\quad%20\eqno{(2)}" />
</div>

The mean square loss of simple linear regression is
<div align="center">
<img src = "http://latex.codecogs.com/gif.latex?L_{reg}%20\left(%20W%20\right)%20=%20\frac%20{%201%20}{%20n%20}%20\sum%20_{%20i=1%20}^{%20n%20}{%20\left(%20y_{%20i%20}%20-%20W%20^{%20T%20}X_{%20i%20}%20\right)%20^{%202%20}%20}%20\quad%20\left(3%20\right)"/>
</div>

The corresponding gradient with respect to <img src = "http://latex.codecogs.com/gif.latex?W" /> in simple linear regression is
 
<div align="center">
<img src = "http://latex.codecogs.com/gif.latex?\frac%20{%20\partial%20L_{%20reg%20}%20}{%20\partial%20W%20}%20=-X^{%20T%20}\left(%20y-XW%20\right)%20\quad%20\left(%204%20\right)" />
</div> 

To minimize the mean square loss Lreg, we can use closed-formed solution or the gradient descent method.

### 2.1. Closed-formed Solution
let <img src = "http://latex.codecogs.com/gif.latex?\frac%20{%20\partial%20L_{%20reg%20}%20}{%20\partial%20W%20}%20=%200" />, we can get

<div align="center">
<img src = "http://latex.codecogs.com/gif.latex?W%20^{%20*%20}=\left(%20X^{%20T%20}X%20\right)%20^{%20-1%20}X^{%20T%20}y%20\quad%20\left(5%20\right)" />
</div>

if the matrix <img src = "http://latex.codecogs.com/gif.latex?X^{T}X" /> is a full-rank matrix or a positive definite matrix, then its **inverse matrix** exists.
Thus we can use the equation (5) to calculate the best weight vector <img src = "http://latex.codecogs.com/gif.latex?W^*" />.

### 2.2. Gradient Descent Method
However, in most cases the inverse matrix of a given matrix may not exist.
So the closed-form solution can't always work. Gracefully, gradient descent can help.

**Gradient Descent (GD)** tries to minimize the loss function by updating weight vector to minimize the learning rate <img src="http://latex.codecogs.com/gif.latex?\eta" /> muplitying the correspondent gradient with respect to weighted vector in the loss function.

<div align="center">
<img src = "http://latex.codecogs.com/gif.latex?W=W-\eta\frac{\partial%20L_{reg}}{\partial%20W}\quad\left(6\right)" />
</div>

In our linear regression model, it looks like this:

<div align="center">
<img src = "http://latex.codecogs.com/gif.latex?W%20=%20W%20+%20\eta%20X^{%20T%20}\left(%20y-XW%20\right)%20\quad%20\left(7\right)" />
</div>

With regularization, the loss function (3) can be changed into the objective function
<div align="center">
    <img src = "http://latex.codecogs.com/gif.latex?L_{reg}%20\left(%20W%20\right)%20=%20\frac{\lambda}{2}\left\|%20W%20\right\|_{2}^{2}%20+%20\frac%20{%201%20}{%20n%20}%20\sum%20_{%20i=1%20}^{%20n%20}{%20\left(%20y_{%20i%20}%20-%20W%20^{%20T%20}X_{%20i%20}%20\right)%20^{%202%20}%20}%20\quad%20\left(8%20\right)"/>
</div>

Then equation (7) becomes
<div align="center">
    <img src = "http://latex.codecogs.com/gif.latex?W%20=%20\left(1-\lambda\eta%20\right)%20W%20+%20\eta%20X^{%20T%20}\left(%20y-XW%20\right)%20\quad%20\left(9\right)" />
</div>

<!--
### 2.3. Comparison with Closed-form Solution and Gradient Descent Method 

||Closed-form Solution|Gradient Descent Method|
|:-:|:-:|:-:|
|Advantages|Mathematic simplication|Easy to perform|
|Disadvantages|Cannot always work<br>Calculating the inverse matrix is low efficent and consums a lot of time|
-->

## 3.Experiment

### 3.1. Dataset
In this experiment, to perform linear regression we uses [housing_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing) in [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), including 506 samples and each sample has 13 features. The dataset is then divided into train set and validation set.

### 3.2. Experiment Step

#### 3.2.1. closed-form solution of Linear Regression
1. Load the housing_scale dataset and divide it into training set and validation set.
2. Initialize linear model parameters. Set all parameter into zero, initialize it randomly or with normal distribution.
3. Select the mean square loss as the loss function and calculate mean square loss of the training set with the weight vector, denoted as **Loss**.
4. Use the formula of the closed-form solution (5) to get the best weighted vector.
5. Get the **Loss**, **Loss_train** under the training set and **Loss_val**  by validating under validation set and output them.

##### Core Code of closed-form solution

```python
import numpy as np
from sklearn.metrics import mean_squared_error
def linear_reg_closed_form(X_train, y_train, X_val, y_val):
    '''Use the closed-form solution to optimize simple linear regression.
    Attention: This function may not work because the inverse of a given 
            matrix may not exist.

    Parameters
    ----------
    X_train: array-like of shape = (n_train_samples, n_features)
        Samples of training set.

    y_train: array-like of shape = (n_train_samples, 1)
        Groundtruth labels of training set.

    X_val: array-like of shape = (n_val_samples, n_features)
        Samples of validation set.

    y_val: array-like of shape = (n_val_samples, 1)
        Groundtruth labels of validation set.

    Returns
    -------
    w: array-like of shape = (n_features, 1)
        The weight vector.

    b: int
        The bias of linear regression.

    losses_dict: dict
        A dict containing losses evaluated before and after training

    '''
    n_features = X_train.shape[1]

    # make all X[i, 0] = 1
    X_train = np.hstack((np.ones(y_train.shape), X_train))
    X_val = np.hstack((np.ones(y_val.shape), X_val))

    # init weight vector
    w = np.zeros((n_features + 1, 1))  # zero based weight vector
    # w = np.random.random((n_features+1, 1)) # initialize with small random values
    # w = np.random.normal(1, 1, size=(n_features+1, 1))

    losses_dict = {}
    losses_dict['losses_train_origin'] = mean_squared_error(
        y_true=y_train, y_pred=np.dot(X_train, w))
    losses_dict['losses_val_origin'] = mean_squared_error(
        y_true=y_val, y_pred=np.dot(X_val, w))

    # use closed-form solution to update w
    # @ operation equals to np.dot
    try:
        w = np.mat(X_train.T @ X_train).I.getA() @ X_train.T @ y_train
    except Exception as ex:
        print('The inverse of the matrix X_train.T @ X_train doesn\'t exist.')
        print(ex)

    losses_dict['losses_train'] = mean_squared_error(y_train, np.dot(
        X_train, w))
    losses_dict['losses_val'] = mean_squared_error(y_val, np.dot(X_val, w))

    w, b = w[1:, ], w[0, 0]

    return w, b, losses_dict

```

#### 3.2.2. Gradient Descent
1. Load and divide dataset.
2. Initialize linear model parameters. Set all parameter into zero, initialize it randomly or with normal distribution.
3. Choose mean square loss as the loss function. 
4. Calculate the gradient with respect to weight in the objective funtion from each example using equation (8). Denote the opposite direction of gradient as D.
5. Update model: <img src="http://latex.codecogs.com/gif.latex?W_t=W_{t-1}+\eta%20D"/>.
6. Get the loss **loss_train** under the training set and **loss_val** by validating under validation set.
7. Repeate step 4 to 6 for several times, and use the values of **loss_train** and **loss_val** to plot the loss graph. 

##### Core Code of Gradient Descent

```python
import numpy as np
from sklearn.metrics import mean_squared_error
def linear_reg_GD(X_train,
                  y_train,
                  X_val,
                  y_val,
                  max_epoch=200,
                  learning_rate=0.01,
                  penalty_factor=0.5):
    '''Use the gradient descent method to solve simple linear regression.

    Parameters
    ----------
    X_train, X_val : array-like of shape = (n_train_samples, n_features) and (n_val_samples, n_features)
        Samples of training set and validation set.

    y_train : array-like of shape = (n_train_samples, 1) and (n_val_samples, 1) respectively
        Groundtruth labels of training set and validation set.

    max_epoch : int
        The max epoch for training.

    learning_rate : float
        The hyper parameter to control the velocity of gradient descent process, 
        also called step_size.

    penalty_factor : float
        The L2 regular term factor for the objective function.

    Returns
    -------
    w: array-like of shape = (n_features, 1)
        The weight vector.

    b: int
        The bias of linear regression.

    losses_dict: dict
        A dict containing losses evaluated before and after training

    '''
    n_features = X_train.shape[1]

    # make all X[i, 0] = 1
    X_train = np.hstack((np.ones(y_train.shape), X_train))
    X_val = np.hstack((np.ones(y_val.shape), X_val))

    # init weight vector
    w = np.zeros((n_features + 1, 1))  # zero based weight vector
    # w = np.random.random((n_features+1, 1)) # initialize with small random values
    # w = np.random.normal(1, 1, size=(n_features+1, 1))

    losses_train, losses_val = [], []

    for epoch in range(0, max_epoch):
        d = -penalty_factor * w + X_train.T @ (y_train - X_train @ w)
        w += learning_rate * d

        # update learning rate if necessary
        # learning_rate /= (epoch + 1) # emmm...no so good
        # learning_rate /= 1 + learning_rate * penalty_factor * (epoch + 1)

        loss_train = mean_squared_error(
            y_true=y_train, y_pred=np.dot(X_train, w))
        loss_val = mean_squared_error(y_true=y_val, y_pred=np.dot(X_val, w))
        losses_train.append(loss_train)
        losses_val.append(loss_val)

    w, b = w[1:, ], w[0, 0]

    losses_dict = {'losses_train': losses_train, 'losses_val': losses_val}

    return w, b, losses_dict
```

### 3.3. Experiment Result

#### 3.3.1. Ouput Results of the closed-form solution 
For this small dataset with 13 features and 506 samples, the closed-form solution can easily and quickly calculate the desired weight vector and generate output results.

>
    closed-form solution for linear regression
    losses_train_origin = 605.93385
	  losses_val_origin = 23.476533
	       losses_train = 23.476533
	         losses_val = 18.176029

#### 3.3.2. Result of the gradient descent
With carefully selecting suitable hyper parameters learning_rate and penalty_factor,
the gradient descent method can minimize the mean square loss in a certain number of epoches
to a low level.

<img src="img/lab1_GD.png"/>

##### Tuning the learning rate
In the gradient descent method, the learning rate is a important parameter to control the velocity of the gradient descent process, 
which has a large impact on convergence. The following two graphes show that:
* Too large η causes oscillatory and may even diverge
* Too small η makes it too slow to converge

<img src="img/bad learning rate.png"/>
<img src="img/learning_rates.png"/>

We can use the adaptive learning rate:
1. Set larger learning rate at the begining
2. Use relatively smaller learning rate in the later epochs
3. Decrease the learning rate: <img src="http://latex.codecogs.com/gif.latex?\eta%20_{t+1}%20=%20\frac{\eta}{t+1}"/>

## 4.Conclusion
In this report, we manage to perform a simple linear regression simulation based on a small dataset, 
using both the closed-form solution and the gradient descent method. Then we do serveral simple experiments to estimate
their performance and in addition to learn to tune the hyper parameter learning rate.  

## Reference
1. Basic Concepts about Machine Learning: Regression and Optimization, Prof. Mingkui Tan
2. [linear_regression, Wikipidea](https://en.wikipedia.org/wiki/Linear_regression)

