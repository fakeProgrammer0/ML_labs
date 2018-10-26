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
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score

def preprocess(dataset_url, n_features):
    X, y = load_svmlight_file(dataset_url, n_features=n_features)

    y = y.reshape(-1, 1)
    # y = y.reshape(-1, 1)[:, 0] # change y into a 1D ndarray

    X = X.toarray()
    X = np.hstack((np.ones(y.shape), X))

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

    def sign_col_vector(a):
        n = a.shape[0]
        for i in range(0, n):
            a[i][0] = sign(a[i][0])
        return a

    n_train_samples, n_features = X_train.shape

    if batch_size > n_train_samples:
        batch_size = n_train_samples

    # init weight vector
    w = np.ones((n_features, 1))

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

        w = (1 - reg_param * learning_rate) * w - learning_rate * penalty_factor / batch_size * temp_sum

        loss_train = hinge_loss(X_train, y_train, w)
        loss_val = hinge_loss(X_val, y_val, w)
        losses_train.append(loss_train)
        losses_val.append(loss_val)
        print("epoch [%3d]: loss_train = [%.6f]; loss_val = [%.6f]" % (epoch, loss_train, loss_val))

        y_train_predict = sign_col_vector(np.dot(X_train, w)).reshape(n_train_samples)
        y_val_predict = sign_col_vector(np.dot(X_val, w)).reshape(X_val.shape[0])
        f1_train = f1_score(y_true=y_train, y_pred=y_train_predict)
        f1_val = f1_score(y_true=y_val, y_pred=y_val_predict)
        f1_scores_train.append(f1_train)
        f1_scores_val.append(f1_val)

        print("epoch [%3d]: f1_train = [%.6f]; f1_val = [%.6f]\n" % (epoch, f1_train, f1_val))

    return w, losses_train, losses_val, f1_scores_train, f1_scores_val

def hinge_loss(X, y, w):
    return np.average(np.maximum(np.ones(y.shape) - y * np.dot(X, w), np.zeros(y.shape)), axis=0)[0]

def run_svm():
    X_train, y_train = preprocess(train_dataset_url, n_features)
    X_val, y_val = preprocess(val_dataset_url, n_features)
    w, losses_train, losses_val, f1_scores_train, f1_scores_val = svm_SGD(X_train, y_train, X_val, y_val, max_epoch=200, batch_size=100, learning_rate=0.1)

    plt.figure(figsize=(16, 9))
    plt.plot(losses_train, '-', color='r', label='losses_train')
    plt.plot(losses_val, '-', color='b', label='losses_val')
    plt.xlabel('epoch')
    plt.ylabel('hinge_loss')
    plt.legend()
    plt.title('loss graph of svm')
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(f1_scores_train, '-', color='r', label='f1_scores_train')
    plt.plot(f1_scores_val, '-', color='b', label='f1_scores_val')
    plt.xlabel('epoch')
    plt.ylabel('f1_score')
    plt.legend()
    plt.title('f1_scores graph of svm')
    plt.show()

from sklearn.svm import SVC
def check_svm():
    X_train, y_train = preprocess(train_dataset_url, n_features)
    X_val, y_val = preprocess(val_dataset_url, n_features)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    y_train_predict = clf.predict(X_train)
    y_val_predict = clf.predict(X_val)
    f1_train = f1_score(y_true=y_train, y_pred=y_train_predict)
    f1_val = f1_score(y_true=y_val, y_pred=y_val_predict)
    print("f1_train = [%.6f]; f1_val = [%.6f]\n" % (f1_train, f1_val))

if __name__ == "__main__":
    run_svm()
    # check_svm()