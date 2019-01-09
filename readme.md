# Labs of the course [Machine Learning] in SCUT
华工软院机器学习（2018秋）课程实验。在华工，你甚至可以像机器一样地学习

## 1.实验内容

|实验名称|实验要求|实验报告|
|:-|:-:|:-:|
|lab1 Linear Regression and Stochastic Gradient Descent|[lab1 spec](https://www.zybuluo.com/liushiya/note/1301605?tdsourcetag=s_pctim_aiomsg) | [lab1 report](lab1/readme.md) |
|lab2 Logistic Regression and Support Vector Machine |[lab2 spec](https://www.zybuluo.com/liushiya/note/1303225) | [lab2 report](lab2/readme.md)|
|lab3 Face Detection Based on AdaBoost Algorithm|[lab3 spec](https://www.zybuluo.com/liushiya/note/1305548) | [lab3 report](lab3/readme.md)
|lab4 Recommender System Based on Matrix Decomposition|[lab4 spec](https://www.zybuluo.com/liushiya/note/1338003)|[lab4 report](lab4/readme.md)
|lab5 Face Detection Based on Neural Network|[lab5 spec](https://www.zybuluo.com/liushiya/note/1343370) | [lab5 report](lab5/readme.md)
|lab6 XGBoost Experiment Manual|[lab6 spec](https://www.zybuluo.com/liushiya/note/1340092) | [lab6 report](lab6/readme.md)

[注] lab5和lab6的内容会尽快更新完_(•̀ω•́ 」∠)_

<!-- 
### lab1 Linear Regression and Stochastic Gradient Descent
[spec](https://www.zybuluo.com/liushiya/note/1301605?tdsourcetag=s_pctim_aiomsg) / [report](lab1/readme.md) 
### lab2 Logistic Regression and Support Vector Machine
[spec](https://www.zybuluo.com/liushiya/note/1303225) / [report](lab2/readme.md)
### lab3 Face Detection Based on AdaBoost Algorithm
[spec](https://www.zybuluo.com/liushiya/note/1305548) / [report](lab3/readme.md)
### lab4 Recommender System Based on Matrix Decomposition
[spec](https://www.zybuluo.com/liushiya/note/1338003) / [report](lab4/readme.md)
### lab5 Face Detection Based on Neural Network
[spec](https://www.zybuluo.com/liushiya/note/1343370) / [report](lab5/readme.md)
### lab6 XGBoost Experiment Manual
[spec](https://www.zybuluo.com/liushiya/note/1340092) / [report](lab6/readme.md)
-->

## 2.说明
实验内容来自华工软院谭明奎和吴庆耀老师的机器学习课程，也感谢编写实验文档的SAIL师兄师姐~

### 2.1.开源声明
该仓库的所有代码和实验报告全部开源，目的是为了方便学习交流（而不是给学弟学妹们更好地抄作业……）。积淀下来的代码和文档，也记录了我作为一个菜鸟机器学习入门（入坑）的心路历程，以及一些踩过的坑_(:з」∠)_

### 2.2.代码风格
仓库里所有代码的编程语言都采用python。尽管python作为一种脚本语言，书写较随意，可以“怎么方便怎么来”，但我还是更倾向于采用函数式的封装，把机器学习算法封装成函数的形式，这样不仅提高了代码可读性，也增加了代码复用的可能性。函数的文档注释风格如下所示，模仿了numpy的注释文档，个人表示这种文档书写风格很赞~

```python
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

    pass
```







