'''using svm with SGD to solve linear classification
dataset: a9a from LIBSVM Data
'''

# 1.load dataset online
# import requests
# from io import BytesIO
# r = requests.get('''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a''')
# train_dataset_url = BytesIO(r.content)
#
# r = requests.get('''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t''')
# val_dataset_url = BytesIO(r.content)

# 2.load local dataset
train_dataset_url = '../dataset/a9a.txt'
val_dataset_url = '../dataset/a9a_t.txt'

n_features = 123

# --------------------------

import numpy as np
import matplotlib.pyplot as plt
import random
import copy

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def preprocess(dataset_url, n_features):
    dataset_url = copy.deepcopy(dataset_url)
    X, y = load_svmlight_file(dataset_url, n_features=n_features)

    # change y from a 1D ndarray into a column vector
    y = y.reshape(-1, 1)
    X = X.toarray()
    X = np.hstack((np.ones(y.shape), X))

    return X, y

# 该数据集类别不平衡，+1 : -1 = 1 : 3
# 如果直接使用以下方法训练，会使得最终分类结果都为负类，这时hingeloss最小，但是f1_score为0，并不是一个好的分类器
def svm_MSGD(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001,
             learning_rate_lambda=0, penalty_factor_C=0.3):
    ''''set up a SVM model with soft margin method using mini-batch stochastic gradient descent
    :param X_train: train data, a (n_samples, n_features + 1) ndarray, where the 1st column are all ones,
    ie.numpy.ones(n_samples)
    :param y_train: labels, a (n_samples, 1) ndarray
    :param X_val: validation data
    :param y_val: validation labels
    :param max_epoch: the max epoch for training
    :param learning_rate: the hyper parameter to control the velocity of gradient descent process,
    also called step_size
    :param learning_rate_lambda: the regualar term for adaptively changing learning_rate
    :param penalty_factor_C: the penalty factor, which emphases the importance of the loss caused
    by samples in the soft margin
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

        y_train_predict = sign_col_vector(np.dot(X_train, w), threshold=0,
                                          sign_thershold=1).reshape(n_train_samples)
        y_val_predict = sign_col_vector(np.dot(X_val, w), threshold=0, sign_thershold=1).reshape(
            X_val.shape[0])
        f1_train = f1_score(y_true=y_train, y_pred=y_train_predict)
        f1_val = f1_score(y_true=y_val, y_pred=y_val_predict)
        f1_scores_train.append(f1_train)
        f1_scores_val.append(f1_val)

        print("epoch [%3d]: f1_train = [%.6f]; f1_val = [%.6f]" % (epoch, f1_train, f1_val))
        print('confusion matrix of train\n', confusion_matrix(y_true=y_train, y_pred=y_train_predict))
        print('confusion matrix of val\n', confusion_matrix(y_true=y_val, y_pred=y_val_predict), '\n')

    return w, losses_train, losses_val, f1_scores_train, f1_scores_val

def hinge_loss(X, y, w):
    return np.average(np.maximum(np.ones(y.shape) - y * np.dot(X, w), np.zeros(y.shape)), axis=0)[0]

def sign(a, threshold=0, sign_thershold=0):
    # the number of positive labels is much smaller than that of the negative labels
    # it's an imbalance classification problem
    if a > threshold:
        return 1
    elif a == threshold:
        return sign_thershold
    else:
        return -1

def sign_col_vector(a, threshold=0, sign_thershold=0):
    a = copy.deepcopy(a)
    n = a.shape[0]
    for i in range(0, n):
        a[i][0] = sign(a[i][0], threshold, sign_thershold)
    return a

# 该数据集类别不平衡，+1 : -1 = 1 : 3，所以把惩罚系数C拆解成C+和C-，增加C+的权重，使得正类被分类错误的损失增大
def svm_MSGD_imbalance(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, learning_rate_lambda=0, pos_C=1, neg_C=1):
    ''''set up a SVM model with soft margin method using mini-batch stochastic gradient descent
    :param X_train: train data, a (n_samples, n_features + 1) ndarray, where the 1st column are all ones, ie.numpy.ones(n_samples)
    :param y_train: labels, a (n_samples, 1) ndarray
    :param X_val: validation data
    :param y_val: validation labels
    :param max_epoch: the max epoch for training
    :param learning_rate: the hyper parameter to control the velocity of gradient descent process, also called step_size
    :param learning_rate_lambda: the regualar term for adaptively changing learning_rate
    :param pos_C: the penalty factor the positive class
    :param neg_C: the negative factor the negative class
    :return w: the SVM weight vector
    :return losses_train, losses_val: the hinge training / validation loss during each epoch
    :return f1_scores_train, f1_scores_val: the f1_score during each epoch
    '''

    n_train_samples, n_features = X_train.shape

    if batch_size > n_train_samples:
        batch_size = n_train_samples

    # init weight vector
    # w = np.zeros((n_features, 1))
    # w = np.random.random((n_features, 1))
    # w = np.random.normal(1, 1, (n_features, 1))
    # w = np.random.randint(-1, 2, size=(n_features, 1))
    w = np.random.randint(-100, 101, size=(n_features, 1)) / 100

    losses_train = []
    losses_val = []

    f1_scores_train = []
    f1_scores_val = []

    for epoch in range(0, max_epoch):
        sample_indice = random.sample(range(0, n_train_samples), batch_size)
        temp_pos_sum = np.zeros(w.shape)
        temp_neg_sum = np.zeros(w.shape)
        pos_sample_cnt = 0
        neg_sample_cnt = 0
        for i in sample_indice:
            if 1 - y_train[i][0] * np.dot(X_train[i], w)[0] > 0:
                # if 1 - np.dot(X_train[i], w)[0] > 0:
                if y_train[i][0] > 0:
                    temp_pos_sum += -y_train[i][0] * X_train[i].reshape(-1, 1)
                    pos_sample_cnt += 1
                else:
                    temp_neg_sum += -y_train[i][0] * X_train[i].reshape(-1, 1)
                    neg_sample_cnt += 1

        w = (1 - learning_rate) * w - learning_rate * (pos_C * temp_pos_sum + neg_C * temp_neg_sum)

        # update learning_rate
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

# f1 稳定性很差，要得到一个f1比较高，同时loss又比较小的训练模型
def run_svm():
    X_train, y_train = preprocess(train_dataset_url, n_features)
    X_val, y_val = preprocess(val_dataset_url, n_features)

    max_epoch = 200
    batch_size = 512
    learning_rate = 0.001
    pos_C = 2.4
    neg_C = 0.8

    # w, losses_train, losses_val, f1_scores_train, f1_scores_val = svm_MSGD(X_train, y_train, X_val, y_val, max_epoch=200, batch_size=512, learning_rate=0.001, penalty_factor_C=1)
    w, losses_train, losses_val, f1_scores_train, f1_scores_val = svm_MSGD_imbalance(X_train, y_train, X_val, y_val, max_epoch=200, batch_size=batch_size, learning_rate=learning_rate, pos_C=pos_C, neg_C=neg_C)

    plt.figure(figsize=(16, 9))
    plt.plot(losses_train, '-', color='r', label='losses_train')
    plt.plot(losses_val, '-', color='b', label='losses_val')
    plt.xlabel('epoch')
    plt.ylabel('hinge_loss')
    plt.legend()
    plt.title('loss graph of svm\nbatch_size = %d\nlearning_rate = %.f\npos_C = %.2f;neg_C = %.2f' % (batch_size, learning_rate, pos_C, neg_C))
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(f1_scores_train, '-', color='r', label='f1_scores_train')
    plt.plot(f1_scores_val, '-', color='b', label='f1_scores_val')
    plt.xlabel('epoch')
    plt.ylabel('f1_score')
    plt.legend()
    plt.title('f1_scores graph of svm\nbatch_size = %d\nlearning_rate = %.f\npos_C = %.2f;neg_C = %.2f' % (batch_size, learning_rate, pos_C, neg_C))
    plt.show()

# check 看下sklearn中svm的实现里f1_score怎样？
from sklearn.svm import SVC
def check_svm():
    def preprocess_svm(dataset_url, n_features):
        X, y = load_svmlight_file(dataset_url, n_features=n_features)
        X = X.toarray()
        return X, y

    X_train, y_train = preprocess_svm(train_dataset_url, n_features)
    X_val, y_val = preprocess_svm(val_dataset_url, n_features)

    # 方法1
    # clf = SVC(gamma='auto')

    # 方法2 设置权重
    # C_pos = np.int(np.sum(y_train==-1) / np.sum(y_train==1))
    # clf = SVC(kernel='linear', class_weight={1:C_pos})

    # 方法3 让SVM自动调节权重
    clf = SVC(kernel='linear', class_weight='balanced')

    clf.fit(X_train, y_train)

    y_train_predict = clf.predict(X_train)
    y_val_predict = clf.predict(X_val)
    f1_train = f1_score(y_true=y_train, y_pred=y_train_predict)
    f1_val = f1_score(y_true=y_val, y_pred=y_val_predict)
    print("f1_train = [%.6f]; f1_val = [%.6f]\n" % (f1_train, f1_val))

    print('confusion matrix of train\n', confusion_matrix(y_true=y_train, y_pred=y_train_predict))
    print('confusion matrix of val\n', confusion_matrix(y_true=y_val, y_pred=y_val_predict), '\n')

if __name__ == "__main__":
    run_svm()
    # check_svm()