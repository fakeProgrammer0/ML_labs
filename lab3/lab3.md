# Face Detection Based on AdaBoost Algorithm

## Abstract
**AdaBoost** is one of the most classic **Boosting** methods. 
In this report, we will try to solve a face detection problem based on a small dataset using AdaBoost. 
A few theory and methodology of **AdaBoost** will be exhibited, followed by several experiments.  

## 1.Introduction
**Boosting** is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones.[7]

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
When adapted to the binary classification problems, the AdaBoost model changes to:

<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?H\left(X_i\right)=sign\left(\sum_{m=1}^{T}{\alpha_mh_m\left(X_i\right)}\right)\quad\left(3\right)"
    />
</div>

Using the **Reweighting** method, AdaBoost tries to increase weights of those samples misclassified in the previous training epoches while decrease weights of samples classified correctly.

Let <img src="http://latex.codecogs.com/gif.latex?\varepsilon_m"/> be the error rate of the base learner <img src="http://latex.codecogs.com/gif.latex?h_m\left(X\right)"/> at epoch <img src="http://latex.codecogs.com/gif.latex?m"/>, then <img src="http://latex.codecogs.com/gif.latex?\alpha_m"/> can be evaluated by:
 
 <div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?\alpha_m=\frac{1}{2}\ln\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)\quad\left(4\right)"
    />
</div>
 
The weighting vector <img src="http://latex.codecogs.com/gif.latex?\omega"/> is updated using:
 
<div align="center">
    <img src=
    "http://latex.codecogs.com/gif.latex?\omega\left(m+1,i\right)=\frac{\omega\left(m,i\right)e^{-\alpha_my_ih_m\left(X_i\right)}}{Z_{m}}\quad\left(5\right)"
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

 <img src="http://latex.codecogs.com/gif.latex?\omega\left(m+1,i\right)=\frac{\omega\left(m,i\right)e^{-\alpha_my_ih_m\left(X_i\right)}}{Z_{m}}"/>

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

#### 3.3.1. Result of Face Classification
The result of face classification are written into the file report.txt according to **3.2.2 step 4**.

The report generated by training a small dataset with 750 samples.

    1.loss estimate of a single weak classifier (a sklearn.tree.DecisionTreeClassifier with max_depth = 1):
    timestamp: 2018-11-16 21:12:25.321301
    
    train_loss_exp = 567.359477
    train_loss_01  = 0.165333
    val_loss_exp   = 204.789175
    val_loss_01    = 0.192000
    
    classification_report of train data:
                  precision    recall  f1-score   support
    
            face       0.88      0.77      0.82       369
        non-face       0.80      0.90      0.85       381
    
       micro avg       0.83      0.83      0.83       750
       macro avg       0.84      0.83      0.83       750
    weighted avg       0.84      0.83      0.83       750
    
    
    classification_report of val data:
                  precision    recall  f1-score   support
    
            face       0.92      0.69      0.79       131
        non-face       0.74      0.93      0.82       119
    
       micro avg       0.81      0.81      0.81       250
       macro avg       0.83      0.81      0.81       250
    weighted avg       0.83      0.81      0.81       250
    
    
    2.loss estimate of AdaBoost (base classifier: sklearn.tree.DecisionTreeClassifier with max_depth = 1):
    timestamp: 2018-11-16 21:15:44.762205
    
    train_loss_exp = 318.216824
    train_loss_01  = 0.024000
    val_loss_exp   = 124.875494
    val_loss_01    = 0.056000
    
    classification_report of train data:
                  precision    recall  f1-score   support
    
            face       0.98      0.97      0.98       369
        non-face       0.97      0.98      0.98       381
    
       micro avg       0.98      0.98      0.98       750
       macro avg       0.98      0.98      0.98       750
    weighted avg       0.98      0.98      0.98       750
    
    
    classification_report of val data:
                  precision    recall  f1-score   support
    
            face       0.96      0.93      0.95       131
        non-face       0.93      0.96      0.94       119
    
       micro avg       0.94      0.94      0.94       250
       macro avg       0.94      0.94      0.94       250
    weighted avg       0.94      0.94      0.94       250

From the report we can see that by using the AdaBoost method, we can combine weak classifiers to get a better classified performance.
 
#### 3.3.2. Result of Face Detection

<img src='img/face_detect.png'/>

#### *3.3.3. Loss Estimate During AdaBoost Training




## 4.Conclusion
In this report, we learn about the methodology about AdaBoost.
Then we perform a face classification problem using AdaBoost.
This report further estimates loss of the dataset during AdaBoost training process.

## References
1. [Face Detection Based on AdaBoost Algorithm](https://www.zybuluo.com/liushiya/note/1305570)
2. Prof.Mingkui Tan. "Ensemble.ppt"
3. Prof. Mingkui Tan. "Boosting Method(AdaBoost, GBDT and XGBoost).pdf"
4. 周志华《机器学习》
5. 李航《统计学习方法》
6. Wikipedia. AdaBoost[https://en.wikipedia.org/wiki/AdaBoost]
7. Wikipedia. Boosting[https://en.wikipedia.org/wiki/Boosting_(machine_learning)]














