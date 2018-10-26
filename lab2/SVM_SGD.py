'''
using svm with SGD to solve linear classification
'''



# ------------
# train_dataset_url = '../dataset/a9a.txt'
# val_dataset_url = '../dataset/a9a_t.txt'
train_dataset_url = r'D:\MyData\Temp Codes\Github\ML_labs\dataset\a9a.txt'
val_dataset_url = r'D:\MyData\Temp Codes\Github\ML_labs\dataset\a9a_t.txt'

n_features = 123

# --------------------------

import numpy as np
import math
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_svmlight_file

def preprocess(dataset_url, n_features):
    X, y = load_svmlight_file(dataset_url, n_features=n_features)

    X = X.toarray()
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    y = y.reshape(-1, 1)
    # y = y.reshape(-1, 1)[:, 0] # change y into a 1D ndarray

    return X, y

def svm_SGD(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, reg_param=0.5, penalty_factor=0.3):
    '''

    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param batch_size:
    :param max_epoch:
    :param learning_rate:
    :return:
    '''

    def sign(a, threshold=0):
        if a > threshold:
            return 1
        else:
            return -1

    n_train_samples, n_features = X_train.shape

    if batch_size > n_train_samples:
        batch_size = n_train_samples

    # init weight vector
    w = np.ones((n_features, 1))

    losses_train = []
    losses_val = []

    for epoch in range(0, max_epoch):
        sample_indice = random.sample(range(0, n_train_samples), batch_size)
        temp_sum = np.zeros(w.shape)
        for i in sample_indice:
            if 1 - y_train[i][0] * np.dot(X_train[i], w)[0] > 0:
                temp_sum += -y_train[i][0] * X_train[i].reshape(-1, 1)

        w = (1 - reg_param) * w - penalty_factor / batch_size * temp_sum

        loss_train = hinge_loss(X_train, y_train, w)
        loss_val = hinge_loss(X_val, y_val, w)

        losses_train.append(loss_train)
        losses_val.append(losses_val)

        print("epoch [%4d]: loss_train = [%.6f]; loss_val = [%.6f]" % (epoch, loss_train, loss_val))

    return w, losses_train, losses_val

def hinge_loss(X, y, w):
    return np.average(np.maximum(np.ones(y.shape) - y * np.dot(X, w), np.zeros(y.shape)), axis=0)

def run_svm():
    X_train, y_train = preprocess(train_dataset_url, n_features)
    X_val, y_val = preprocess(val_dataset_url, n_features)
    w, losses_train, losses_val = svm_SGD(X_train, y_train, X_val, y_val)

    plt.figure(figsize=(16, 9))
    plt.plot(losses_train, '-', color='r', label='losses_train')
    plt.plot(losses_val, '-', color='b', label='losses_val')
    plt.xlabel('epoch')
    plt.ylabel('hinge_loss')
    plt.legend()
    plt.title('loss graph of svm')
    plt.show()

if __name__ == "__main__":
    run_svm()