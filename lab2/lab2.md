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

## 4.Conclusion


## References



instead of <br/> 
<img src="http://latex.codecogs.com/gif.latex?W=W-\eta\frac{\partial%20J\left(W\right)}{\partial%20W}=\left(1-\eta\lambda\right)W+\eta\frac{1}{n}\sum_{i=1}^{n}{\frac{y_iX_i}{1+e^{y_iW^TX_i}}}\quad\left(13\right)"/>

