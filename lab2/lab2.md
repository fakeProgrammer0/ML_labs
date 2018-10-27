# Logistic Regression and Support Vector Machine

## Abstract
In this report, we will solve **binary classification** using both **logistic regression** and **support vector machine (SVM)**.


## 1.Introduction
**Binary Classification** is one of the most fundamental problems in machine learning.
**Logistic Regression (LR)** and **Support Vector Machine (SVM)** are two efficent and powerful tools to solve binary classification.
Logistic regression try to model a function to predict the probility of a sample to be in the postive or negative class 
while support vector machine is aimed to find a hyperplane which will seperate the postive samples and negative samples.

In this report, we will first explain the methodology of both logistic regression and support vector machine.
Then a variant of Gradient Descent called **Mini-batch Stochastic Gradient Descent (MSGD)** is used to implement the two above methods.

Motivations of Experiment are listed below:
1. Compare andand understand the difference between gradient descent and batch random stochastic gradient descent.
2. Compare and understand the differences and relationships between Logistic regression and linear classification.
3. Further understand the principles of SVM and practice on larger data.    

## 2.Methods and Theory

### 2.1. Mini-batch Stochastic Gradient Descent
Mini-batch Stochastic Gradient Descent updates weight vector using the gradient with respect to weight in the objective function
but each time it select a mini batch of samples to perform updating instead of using all the samples as gradient descent.

<img src="http://latex.codecogs.com/gif.latex?W_t=W_{t-1}-\frac{\eta}{\left|S_k\right|}%20\sum_{i\in\left\|S_k\right\|}\nabla_WL_i\left(W\right)\quad\left(1\right)"/><br/>

### 2.2. Logistic Regression
The equation of logistic regression can be described as:<br/>
<img src = "http://latex.codecogs.com/gif.latex?y=g\left(w^{T}X+b\right)\quad\eqno{(2)}"/><br/>

where g(z) is the logisitc function, <img src = "http://latex.codecogs.com/gif.latex?g\left(z\right)=\frac{1}{1+e^{-z}}\quad\left(3\right)"/><br/>

Let <img src = "http://latex.codecogs.com/gif.latex?W=\left(b;\omega\right)" />
and then equation (2) can be changed into 
<img src = "http://latex.codecogs.com/gif.latex?y=g\left(W^TX\right)\quad\eqno{(4)}" /><br/>

With equation (3), we can get 
<img src = "http://latex.codecogs.com/gif.latex?y=\frac{1}{1+e^{-W^TX}}\quad\left(5\right)"/><br/>

<img src="http://latex.codecogs.com/gif.latex?\ln\frac{y}{1-y}=W^TX\quad\left(6\right)" />

if we regard **y** in **equation (6)** to be the posterior probability of 
<img src="http://latex.codecogs.com/gif.latex?p\left(y=1|x\right)"/>,
then equation (6) can be transformed into  

<img src="http://latex.codecogs.com/gif.latex?\ln\frac{p\left(y=1|x\right)}{p\left(y=-1|x\right)}=W^TX\quad\left(7\right)"/>

where <img src = "http://latex.codecogs.com/gif.latex?p\left(y=1|x\right)=\frac{1}{1+e^{-W^TX}}\quad\left(8\right)"/>, 
<img src = "http://latex.codecogs.com/gif.latex?p\left(y=-1|x\right)=\frac{1}{1+e^{W^TX}}\quad\left(9\right)"/>

Using the Maximum Log Likelihood Estimate, we can have the objective function  
<img src = "http://latex.codecogs.com/gif.latex?L\left(W\right)=\sum_{i=1}^{n}{p\left(y_i|x_i\right)}\quad\left(10\right)"/>

<img src="img/lab2_LR_MLE.png"/>

After the above induction and regularization, we update our objective funtion to
<div align="center">
<img src="http://latex.codecogs.com/gif.latex?J\left(W\right)=\frac{1}{n}\sum_{i=1}^{n}{\log\left(1+e^{-y_iw^Tx_i}\right)}+\frac{\lambda}{2}{\left\|W\right\|_2^2}\quad\left(11\right)" align="center"/>
</div>

and we can update weight vector using mini-batch gradient descent:<br/>

<img src="http://latex.codecogs.com/gif.latex?W=W-\eta\frac{\partial%20J\left(W\right)}{\partial%20W}=\left(1-\eta\lambda\right)W+\eta\frac{1}{\left|S_k\right|}\sum_{i\in\left|S_k\right|}{\frac{y_iX_i}{1+e^{y_iW^TX_i}}}\quad\left(12\right)"/>

### 2.3. Support Vector Machine
Using **hinge loss** and **soft margin** method, the objective function of SVM is

<img src="http://latex.codecogs.com/gif.latex?L(W)=\frac{\left\|W\right\|_2^2}{2}+\frac{C}{N}\sum_{i=1}^{N}{\max\left(0, 1-y_iW^TX_i\right)}\quad\left(13\right)"/> 

where the gradient with respect to W in the objective function is

<img src="http://latex.codecogs.com/gif.latex?g_W\left(X_i\right)=\begin{cases}-y_iX_i\quad1-y_iW^TX_i>0\\0\quad\quad\quad1-y_iW^TX_i\le0\end{cases}\quad\left(14\right)"/>

Using **MSGD** method, the weight vector can be updated in this way:

<img src="http://latex.codecogs.com/gif.latex?W=W-\eta\frac{\partial%20L\left(W\right)}{\partial%20W}=\left(1-\eta\lambda\right)W+\eta\frac{C}{\left|S_k\right|}\sum_{i\in\left|S_k\right|}{g_W\left(X_i\right)}\quad\left(15\right)"/>

## 3.Experiment

### 3.1. Dataset
In this experiment, to perform binary classification we uses [a9a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a) in [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), including 32561/16281(testing) samples and each sample has 123 features.
a9a_t is a9a's validation dataset.

### 3.2. Experiment Step

### 3.2.1. Logistic Regression
1. Load the training set and validation set.
2. Initialize logistic regression model parameter with zeros, random numbers or normal distribution.
3. Determine the size of the batch_size and randomly take some samples,calculate gradient G toward loss function from partial samples.
4. Use the **MSGD** optimization method described in **equation (12)** to update the parametric model.
5. Predict under validation set and get the loss **L_val** using **equation (11)**.
6. Repeat step 3 to 5 for several times, and drawing graph of **L_val**  with the number of iterations.

<!-- 5. Select the appropriate threshold, mark the sample whose predict scores greater than the threshold as positive, on the contrary as negative. Predict under validation set and get the loss **L_val**. -->

#### Core Code of LR

```python
import numpy as np
import random
import math
def log_reg_MLE_MSGD(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, reg_param=0.3):
    '''logistic regression using mini-batch stochastic gradient descent with maximum likelihood method
    :param X_train: train data, a (n_samples, n_features + 1) ndarray, where the 1st column are all ones, ie.numpy.ones(n_samples)
    :param y_train: labels, a (n_samples, 1) ndarray
    :param X_val: validation data
    :param y_val: validation labels
    :param max_epoch: the max epoch for training
    :param learning_rate: the hyper parameter to control the velocity of gradient descent process, also called step_size
    :param reg_param: the L2 regular term factor for the objective function

    :return w: the weight vector, a (n_features + 1, 1) ndarray
    :return log_LEs_train: the min log likelihood estimate of the training set during each epoch
    :return log_LEs_val: the min log likelihood estimate of the validation set during each epoch
    '''
    n_train_samples, n_features = X_train.shape
    if n_train_samples < batch_size:
        batch_size = n_train_samples

    # for calculation convenience, y is represented as a row vector
    y_train = y_train.reshape(1, -1)[0, :]
    y_val = y_val.reshape(1, -1)[0, :]

    # init weight vectors
    # for calculation convenience, w is represented as a row vector
    w = np.zeros(n_features)

    log_LEs_train = []
    log_LEs_val = []

    for epoch in range(0, max_epoch):

        temp_sum = np.zeros(w.shape)
        batch_indice = random.sample(range(0, n_train_samples), batch_size)

        for idx in batch_indice:
            exp_term = math.exp(-y_train[idx] * np.dot(X_train[idx], w))
            temp_sum += y_train[idx] * X_train[idx] * exp_term / (1 + exp_term)

        # update w using gradient of the objective function
        w = (1 - learning_rate * reg_param) * w + learning_rate / batch_size * temp_sum

        log_LE_train = min_log_LE(X_train, y_train, w)
        log_LEs_train.append(log_LE_train)
        log_LE_val = min_log_LE(X_val, y_val, w)
        log_LEs_val.append(log_LE_val)
        print("epoch {:3d}: loss_train = [{:.6f}]; loss_val = [{:.6f}]".format(epoch, log_LE_train, log_LE_val))

    return w, log_LEs_train, log_LEs_val

def min_log_LE(X, y, w):
    '''The min log likelihood estimate
    :param X: the data, a (n_samples, n_features) ndarray
    :param y: the groundtruth labels, required in a row shape
    :param w: the weight vector, required in a row shape
    '''
    n_samples = X.shape[0]
    loss_sum = 0
    for i in range(0, n_samples):
        loss_sum += np.log(1 + np.exp(-y[i] * (np.dot(X[i], w))))

    return loss_sum / n_samples
```

### 3.2.2. Support Vector Machine
1. Load the training set and validation set.
2. Initialize SVM model parameter with zeros, random numbers or normal distribution.
3. Determine the size of the batch_size and randomly take some samples,calculate gradient G toward loss function from partial samples using **equation(14)**.
4. Use the **MSGD** optimization method described in **equation (15)** to update the parametric model.
5. Predict under validation set and get the loss **L_val** using **hinge loss**.
6. Repeat step 3 to 5 for several times, and drawing graph of **L_val**  with the number of iterations.

#### Core Code of SVM

```python
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def svm_SGD(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, learning_rate_lambda=0, penalty_factor_C=0.3):
    ''''set up a SVM model with soft margin method using mini-batch stochastic gradient descent
    :param X_train: train data, a (n_samples, n_features + 1) ndarray, where the 1st column are all ones, ie.numpy.ones(n_samples)
    :param y_train: labels, a (n_samples, 1) ndarray
    :param X_val: validation data
    :param y_val: validation labels
    :param max_epoch: the max epoch for training
    :param learning_rate: the hyper parameter to control the velocity of gradient descent process, also called step_size
    :param learning_rate_lambda: the regualar term for adaptively changing learning_rate
    :param penalty_factor_C: the penalty factor, which emphases the importance of the loss caused by samples in the soft margin
    :return w: the SVM weight vector
    :return losses_train, losses_val: the hinge training / validation loss during each epoch
    :return f1_scores_train, f1_scores_val: the f1_score during each epoch
    '''
    
    n_train_samples, n_features = X_train.shape

    if batch_size > n_train_samples:
        batch_size = n_train_samples

    # init weight vector
    w = np.zeros((n_features, 1))
    # w = np.random.random((n_features, 1))
    # w = np.random.normal(1, 1, (n_features, 1))
    # w = np.random.randint(-1, 2, size=(n_features, 1))

    losses_train = []
    losses_val = []

    f1_scores_train = []
    f1_scores_val = []

    for epoch in range(0, max_epoch):
        sample_indice = random.sample(range(0, n_train_samples), batch_size)
        temp_sum = np.zeros(w.shape)
        for i in sample_indice:
            if 1 - y_train[i][0] * np.dot(X_train[i], w)[0] > 0:
                temp_sum += -y_train[i][0] * X_train[i].reshape(-1, 1)

        # no regularization
        w = (1 - learning_rate) * w - learning_rate * penalty_factor_C / batch_size * temp_sum

        # update learning_rate if needed
        learning_rate /= 1 + learning_rate * learning_rate_lambda * (epoch + 1)

        loss_train = hinge_loss(X_train, y_train, w)
        loss_val = hinge_loss(X_val, y_val, w)
        losses_train.append(loss_train)
        losses_val.append(loss_val)
        print("epoch [%3d]: loss_train = [%.6f]; loss_val = [%.6f]" % (epoch, loss_train, loss_val))

        y_train_predict = sign_col_vector(np.dot(X_train, w), threshold=0, sign_thershold=1).reshape(n_train_samples)
        y_val_predict = sign_col_vector(np.dot(X_val, w), threshold=0, sign_thershold=1).reshape(X_val.shape[0])
        f1_train = f1_score(y_true=y_train, y_pred=y_train_predict)
        f1_val = f1_score(y_true=y_val, y_pred=y_val_predict)
        f1_scores_train.append(f1_train)
        f1_scores_val.append(f1_val)

        print("epoch [%3d]: f1_train = [%.6f]; f1_val = [%.6f]" % (epoch, f1_train, f1_val))
        print('confusion matrix of train\n', confusion_matrix(y_true=y_train, y_pred=y_train_predict))
        print('confusion matrix of val\n', confusion_matrix(y_true=y_val, y_pred=y_val_predict), '\n')

    return w, losses_train, losses_val, f1_scores_train, f1_scores_val

def sign(a, threshold=0, sign_thershold=0):
    # the number of positive labels is much smaller than that of the negative labels
    # it's an imbalance classification problem
    if a > threshold:
        return 1
    elif a == threshold:
        return sign_thershold
    else:
        return -1

import copy
def sign_col_vector(a, threshold=0, sign_thershold=0):
    a = copy.deepcopy(a)
    n = a.shape[0]
    for i in range(0, n):
        a[i][0] = sign(a[i][0], threshold, sign_thershold)
    return a

def hinge_loss(X, y, w):
    return np.average(np.maximum(np.ones(y.shape) - y * np.dot(X, w), np.zeros(y.shape)), axis=0)[0]
```

### 3.3. Experiment Results



## 4.Conclusion


## References



instead of <br/> 
<img src="http://latex.codecogs.com/gif.latex?W=W-\eta\frac{\partial%20J\left(W\right)}{\partial%20W}=\left(1-\eta\lambda\right)W+\eta\frac{1}{n}\sum_{i=1}^{n}{\frac{y_iX_i}{1+e^{y_iW^TX_i}}}\quad\left(13\right)"/>

