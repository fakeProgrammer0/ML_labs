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


<img src="http://latex.codecogs.com/gif.latex?G=\frac{\partial%20L}{\partial%20W}"/>  

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

<img src="img/lab2_LR_MLE.png"/>



## 3.Experiment

## 4.Conclusion


## References