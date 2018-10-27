# Linear Regression and Stochastic Gradient Descent

## Abstract
In this report, we will solve linear regression using both the closed-form solution and gradient descent method based on a small dataset.
After that, we will further learn to tune some parameters such as the learing rate to optimizate our gradient descent model.  

## I.Introduction
In statistics, linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression.\[[1](linear_regression Wikipidea)\]

Motivations of the report are listed below:
* Further understand of linear regression ，closed-form solution and Stochastic gradient descent.
* Conduct some experiments under small scale dataset.
* Realize the process of optimization and adjusting parameters.

## II.Methods and Theory
The equation of simple linear regression can be described as:<br/>
<img src = "http://latex.codecogs.com/gif.latex?y = w^{T}X + b \quad \eqno{(1)}"/><br/>

Let <img src = "http://latex.codecogs.com/gif.latex?\beta = \left(b;\omega \right) " />
and then equation (1) can be changed into <br/>
<img src = "http://latex.codecogs.com/gif.latex?y = \beta ^T X \quad \eqno{(2)}" /><br/>

The least square loss of simple linear regression is <br/>
<img src = "http://latex.codecogs.com/gif.latex?L_{reg} \left( \beta  \right) = \frac { 1 }{ n } \sum _{ i=1 }^{ n }{ \left( y_{ i } - \beta ^{ T }X_{ i } \right) ^{ 2 } } \quad \left(3 \right) "/>

The corresponding gradient with respect to <img src = "http://latex.codecogs.com/gif.latex?\beta" /> in simple linear regression is
 
<img src = "http://latex.codecogs.com/gif.latex?\frac { \partial L_{ reg } }{ \partial \beta  } =-X^{ T }\left( y-X\beta \right)  \quad \left( 4 \right)" /> 

To minimize the least square loss Lreg, we can use closed-formed solution or the gradient descent method.

### Closed-formed Solution
let <img src = "http://latex.codecogs.com/gif.latex?\frac { \partial L_{ reg } }{ \partial \beta  } = 0" />, we can get

<img src = "http://latex.codecogs.com/gif.latex?\beta ^{ * }=\left( X^{ T }X \right) ^{ -1 }X^{ T }y \quad \left(5 \right)" />

if the matrix <img src = "http://latex.codecogs.com/gif.latex?X^{T}X" /> is a full-rank matrix or a positive definite matrix, then its **inverse matrix** exists.
Thus we can use the equation (5) to calculate the best weight vector <img src = "http://latex.codecogs.com/gif.latex?\beta^*" />.

### Gradient Descent Method
However, in most cases the inverse matrix of a given matrix may not exist.
So the closed-form solution can't always work. Gracefully, gradient descent can help.

**Gradient Descent (GD)** tries to minimize the loss function by updating weight vector to minimize the learning rate <img src="http://latex.codecogs.com/gif.latex?\eta" /> muplitying the correspondent gradient with respect to weighted vector in the loss function.

<img src = "http://latex.codecogs.com/gif.latex?\beta = \beta - \eta\frac{\partial L_{reg}}{\partial \beta} \quad \left(6\right)" />

In our linear regression model, it looks like this:

<img src = "http://latex.codecogs.com/gif.latex?\beta = \beta + \eta X^{ T }\left( y-X\beta \right) \quad \left(7\right)" />

With regularization, the loss function (3) can be changed into the objective function
<br/><img src = "http://latex.codecogs.com/gif.latex?L_{reg} \left( \beta  \right) = \frac{\lambda}{2}\left\| \beta \right\|_{2}^{2} + \frac { 1 }{ n } \sum _{ i=1 }^{ n }{ \left( y_{ i } - \beta ^{ T }X_{ i } \right) ^{ 2 } } \quad \left(8 \right) "/><br/>
Then equation (7) becomes
<br/><img src = "http://latex.codecogs.com/gif.latex?\beta = \left(1-\lambda\eta \right) \beta + \eta X^{ T }\left( y-X\beta \right) \quad \left(9\right)" /><br/>

### Comparison with Closed-form Solution and Gradient Descent Method 

||Closed-form Solution|Gradient Descent Method|
|:-:|:-:|:-:|
|Advantages|Mathematic simplication|Easy to perform|
|Disadvantages|Cannot always work<br>Calculating the inverse matrix is low efficent and consums a lot of time|

latex equation
<img src = "http://latex.codecogs.com/gif.latex?11" />

## III.Experiment

### A.Dataset
In this experiment, to perform linear regression we uses [housing_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing) in [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), including 506 samples and each sample has 13 features. The dataset is then divided into train set and validation set.

### B.Experiment Step

#### closed-form solution of Linear Regression
1. Load the housing_scale dataset and devide it into training set and validation set.
2. Initialize linear model parameters. Set all parameter into zero, initialize it randomly or with normal distribution.
3. Select the least square loss as the loss function and calculate least square loss of the training set with the weight vector, denoted as **Loss**.
4. Use the formula of the closed-form solution (5) to get the best weighted vector.
5. Get the **Loss**, **Loss_train** under the training set and **Loss_val**  by validating under validation set and output them.




>
    closed-form solution for linear regression
	     loss0 = 605.933852
	     loss1 = 23.476533
	loss_train = 23.476533
	  loss_val = 18.176029


## IV.Conclusion

## Reference
1.[linear_regression Wikipidea](https://en.wikipedia.org/wiki/Linear_regression)


<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda \theta_k^{(j)}" />

![ss](http://latex.codecogs.com/gif.latex?\\frac{1}{1+sin(x)})

33 <br/>

![s](http://latex.codecogs.com/gif.latex? y = w^{T}X + b)
