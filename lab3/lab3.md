# Face Detection Based on AdaBoost Algorithm

## Abstract
**AdaBoost** is one of the most classic **Boosting** methods. 
In this report, we will try to solve a face detection problem based on a small dataset using AdaBoost. 
A few theory and methodology of **AdaBoost** will be exhibited, followed by several experiments.  

## 1.Introduction
**Boosting** is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones.[1]

**AdaBoost**, short for **Adaptive Boosting**, is a Boosting method using exponential loss function which emphasize samples classified incorectly in the previous training epoches. 

In this report, we will first explain the methodology of AdaBoost.
Equipped with the powerful tool of AdaBoost, we will solve a face detection problem.  

Motivations of Experiment are listed below:
1. Understand AdaBoost further
2. Get familiar with the basic method of face detection
3. Learn to use AdaBoost to solve the face detection problem, and combine the theory with the actual project
4. Experience the complete process of machine learning   

## 2.Methods and Theory

Here we will briefly introduce some important facts of AdaBoost (rather than a complete whole of the mathematical derivation process or the statistical guarantee proving of AdaBoost).

From the perspective of **additive model**, the AdaBoost model <img src="http://latex.codecogs.com/gif.latex?H\left(X_i\right)"/> can be regarded as a linear composition of many base learners <img src="http://latex.codecogs.com/gif.latex?h_m\left(X_i\right)"/>, where <img src="http://latex.codecogs.com/gif.latex?\alpha_m"/> is the corresponding weight of <img src="http://latex.codecogs.com/gif.latex?h_m\left(X_i\right)"/>.

<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?H\left(X_i\right)=\sum_{m=1}^{T}{\alpha_mh_m\left(X_i\right)}\quad\left(1\right)"
    />
</div>

<br/>

<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?H_m\left(X_i\right)=H_{m-1}\left(X_i\right)+\alpha_mh_m\left(X_i\right)"
    />
</div>

<br/>

AdaBoost use **the exponential loss function** to evaluate and minimize the loss:

<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?L\left(H\left(X\right)\right)=\sum_{i=1}^{N}{e^{-y_iH\left(X_i\right)}}\quad\left(2\right)"
    />
</div>

<br/>

The following derivation **mainly talks about the binary classification problem where the label y is from {-1, +1}**.

Using the **Reweighting** method, AdaBoost tries to increase weights of those samples misclassified in the previous training epoches while decrease weights of samples classified correctly.

Let <img src="http://latex.codecogs.com/gif.latex?\varepsilon_m"/> be the error rate of the base learner <img src="http://latex.codecogs.com/gif.latex?h_m\left(X\right)"/> at epoch <img src="http://latex.codecogs.com/gif.latex?m"/>, then <img src="http://latex.codecogs.com/gif.latex?\alpha_m"/> can be evaluated by:
 
 <div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?\alpha_m=\frac{1}{2}\ln\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)\quad\left(3\right)"
    />
</div>
 
The weighting vector <img src="http://latex.codecogs.com/gif.latex?\omega"/> is updated using:
 
<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?\omega\left(m+1,i\right)=\frac{\omega\left(m,i\right)e^{-\alpha_my_ih_m\left(X_i\right)}}{Z_{m}}\quad\left(4\right)"
    />
</div>
 
where <img src="http://latex.codecogs.com/gif.latex?Z_{m}"/> is the regularization factor:
 
<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?Z_{m}=\sum_{i=1}^{N}{\omega\left(m,i\right)e^{-\alpha_my_ih_m\left(X_i\right)}}"
    />
</div>
 
The pseudocode of AdaBoost can be summarize as:  
 
<div style="background=#000; color:#FFF">

<span>For m = 1, 2, ..., T: </span><br/>
<span>    train </span><img src="http://latex.codecogs.com/gif.latex?h_m\left(X\right)"/> based on the sample weight <img src="http://latex.codecogs.com/gif.latex?\omega_m"/>  
    calculate the error rate <img src="http://latex.codecogs.com/gif.latex?\varepsilon_m"/> of the base learner <img src="http://latex.codecogs.com/gif.latex?h_m\left(X\right)"/>
    
   **if <img src="http://latex.codecogs.com/gif.latex?\varepsilon_m\ge0.5"/> then break**
    
 <img src="http://latex.codecogs.com/gif.latex?\alpha_m=\frac{1}{2}\ln\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)"/>

 <img src="http://latex.codecogs.com/gif.latex?Z_{m}=\sum_{i=1}^{N}{\omega\left(m,i\right)e^{-\alpha_my_ih_m\left(X_i\right)}}"/>

 <img src="http://latex.codecogs.com/gif.latex?\omega\left(m+1,i\right)=\frac{\omega\left(m,i\right)e^{-\alpha_my_ih_m\left(X_i\right)}}{Z_{m}}\quad\left(4\right)"/>

EndFor

return <img src="http://latex.codecogs.com/gif.latex?H\left(X_i\right)=\sum_{m=1}^{T}{\alpha_mh_m\left(X_i\right)}"/>


</div>
 
## 3.Experiment

### 3.1. Dataset
The dataset used in this experiment are from the [example repository](https://github.com/wujiaju/ML2018-lab-03). It provides 1000 pictures, of which 500 are human face RGB images and the other 500 are non-face RGB images. 

### 3.2. Experiment Step

#### 3.2.1. Training procedure of the AdaBoost Model
1. Initialize training set weights <img src="http://latex.codecogs.com/gif.latex?\omega"/>, each training sample is given the same weight <img src="http://latex.codecogs.com/gif.latex?\frac{1}{N}"/>. 
2. Training a base classifier(Here we use a decision tree, **DecisionTreeClassifier**, from **sklearn.tree** library) based on the current sample weights. 
3. Calculate the classification error rate <img src="http://latex.codecogs.com/gif.latex?\varepsilon"/> of the base classifier on the training set. 
4. Calculate the parameter <img src="http://latex.codecogs.com/gif.latex?\alpha"/> according to the classification error rate <img src="http://latex.codecogs.com/gif.latex?\varepsilon"/>. 
5. Update training set weights <img src="http://latex.codecogs.com/gif.latex?\omega"/>. 
6. Repeat steps 2-5 above for iteration. The number of iterations is based on the number of classifiers. 

**Core Code of AdaBoost Training** (written in python)

```python
import math
import numpy as np

def fit(self,X,y):
    '''Build a boosted classifier from the training set (X, y).

    Args:
        X: An ndarray indicating the samples to be trained, 
           which shape should be (n_samples,n_features).
        y: An ndarray indicating the ground-truth labels correspond to X, 
           which shape should be (n_samples,1), 
           where the class label y[i, 0] is from {-1, +1}.
    '''
    w = np.ones(y.shape)
    w = w / w.sum() # regularization

    self.a = []
    self.base_clfs = []

    for i in range(self.n_weakers_limit):
        base_clf = self.weak_clf(max_depth=2)
        base_clf.fit(X, y.flatten(), w.flatten())

        y_pred = base_clf.predict(X).reshape((-1, 1))

        err_rate = w.T.dot(y_pred != y)[0][0] / w.sum()

        if err_rate > 1 / 2 or err_rate == 0.0:
            break

        weight_param_a = math.log((1 - err_rate) / err_rate) / 2

        self.base_clfs.append(base_clf)
        self.a.append(weight_param_a)

        w = w * np.exp(-weight_param_a * y * y_pred)
        w = w / w.sum() # regularization

        # prevent overfiting
        # if self.is_good_enough():
        #     break;

```

#### 3.2.2. Face Classification 
1. Load data set data. The images are converted into grayscale images with size of 24 * 24. Face images are labelled +1 while non-face images are labelled -1.
2. Processing image samples to extract NPD features.
3. The data set is divided into training set and validation set. In this experiment samples of the validation set takes up 25% of the original data set. 
4. Predict and verify the accuracy on the validation set using the method in AdaBoostClassifier and use **classification_report()** of the sklearn.metrics library function writes predicted result to classifier_report.txt.

#### 3.2.3. Face Detection
1. Run the face_detection.py file. Experience the OpenCV's built-in method of face detection using Haar Feature-based Cascade Classifiers. The result will be save as detect_result.jpg.

### 3.3. Experiment Results

#### 3.3.1. Result of Logistic Regressoion
From the following graph, we can see that as the number of epoches increases, the min log likelihood estimate descreases
and then the curve becomes flat and smooth. Thus we can conclude that with the logistic regression method, the min log 
likelihood in **equation (2)** estimate decreases.

<img src="img/lab2_LR_log_likelihood_estimate.png"/>

#### *3.3.2. class imbalance problem
The dataset a9a is an imbalance dataset for the number of positive class is nearly one third of the number of the negative class. 
If the methodology of SVM described in [3.2.2 support vector machine](#3.2.2.-support-vector-machine) is directly used, 
finally we will get a imbalanced classifier which erroneously recognizes all samples to be negative. In this case, the f1_score of 
the classification result is 0. Obviously, this is a good classifier we want.

    epoch [199]: loss_train = [0.624254]; loss_val = [0.618534]
    epoch [199]: f1_train = [0.000000]; f1_val = [0.000000]
    confusion matrix of train
     [[24720     0]
     [ 7841     0]]
    confusion matrix of val
     [[12435     0]
     [ 3846     0]] 

Improvement: Decomposite the penalty factor C into C+ and C- respectly. C+ is the penalty factor of the loss casued by 
the positive samples in the soft margin while C- is the the penalty factor of the loss casued by the nagative samples 
in the soft margin. By increasing the weight of C+, we can emphase the loss caused by the positive samples in the soft margin.
For more details, check the code submitted. 

    epoch [199]: loss_train = [0.460696]; loss_val = [0.457864]
    epoch [199]: f1_train = [0.663994]; f1_val = [0.661774]
    confusion matrix of train
     [[19064  5656]
     [ 1133  6708]]
    confusion matrix of val
     [[9626 2809]
     [ 555 3291]] 

该数据集类别不平衡，+1 : -1 = 1 : 3, 如果直接使用以下方法训练，会使得最终分类结果都为负类，这时hingeloss最小，但是f1_score为0，这并不是一个好的分类器。

改进：把惩罚系数C拆解成C+和C-，增加C+的权重，使得正类被分类错误的损失增大。

#### 3.3.3. Result of Support Vector Machine
After the improvement, the hinge loss decrease and the f1_score increase a lot as the number of epoches grows.  
<img src="img/lab2_SVM_hingeloss.png"/>

<img src="img/lab2_SVM_f1_scores.png"/>


## 4.Conclusion
In this report, we learn about the methodology about the logistic regression and support vector machine.
Then we carry out some experiment on a dataset to estimate the performance of both methods.
This report further examines the class imbalance problem about the SVM model and successfully develop a variant 
of the soft margin SVM to solve the problem.  

## References
1. Linear Classification and Support Vector Machine and Stochastic Gradient Descent, Prof.Mingkui Tan
2. Multi-class Classification and Softmax Regression, Prof.Mingkui Tan
3. [知乎\[ＳＶＭ　当正负样本比例不对称时，调节不同的惩罚参数Ｃ，对结果有什么影响。是不是Ｃ大点，会好点？\]](https://www.zhihu.com/question/56033163/answer/147392242)
4. [SVM: Separating hyperplane for unbalanced classes](https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html%23example-svm-plot-separating-hyperplane-unbalanced-py)














