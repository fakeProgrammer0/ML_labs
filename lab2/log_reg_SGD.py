train_dataset_url = '../dataset/a9a.txt'
val_dataset_url = '../dataset/a9a_t.txt'

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

    return X, y


def log_reg_SGD(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, reg_param=0.3):
    global n_features

    # init weight vectors
    w = np.zeros((n_features + 1, 1))

    n_train_samples = X_train.shape[0]
    if n_train_samples < batch_size:
        batch_size = n_train_samples

    losses_train = []
    losses_val = []

    for epoch in range(0, max_epoch):

        temp_sum = np.zeros((n_features + 1, 1))
        batch_indice = random.sample(range(0, n_train_samples), batch_size)

        for idx in batch_indice:
            temp_sum += X_train[idx] * (y_train[idx][0] - logistic_g(np.dot(X_train[idx], w)[0]))

        # objective function
        # w = (1 - learning_rate * reg_param) * w + learning_rate / batch_size * temp_sum

        # loss function
        w = learning_rate * n_train_samples / batch_size * temp_sum

        # print('w = ', np.floor(w.reshape(1, -1)))

        loss_train = threshold_Ein(X_train, y_train, w)
        losses_train.append(loss_train)

        loss_val = threshold_Ein(X_val, y_val, w)
        losses_val.append(loss_val)

        # print(f"epoch {epoch}: loss_train = {loss_train}; loss_val = {loss_val}")
        print("epoch {}: loss_train = [{:.2f}]; loss_val = [{:.2f}]".format(epoch, loss_train, loss_val))

    return w, losses_train, losses_val


def log_reg_SGD2(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, reg_param=0.3):
    global n_features

    # init weight vectors
    w = np.zeros((n_features + 1, 1))

    n_train_samples = X_train.shape[0]
    if n_train_samples < batch_size:
        batch_size = n_train_samples

    losses_train = []
    losses_val = []

    for epoch in range(0, max_epoch):

        temp_sum = np.zeros((n_features + 1, 1))
        # batch_indice = rnd_sample_indice(0, n_train_samples, batch_size)
        batch_indice = random.sample(range(0, n_train_samples), batch_size)

        for idx in batch_indice:
            # t = y_train[idx][0] * (X_train[idx]).T
            # t /= (1 + math.exp(y_train[idx] * np.dot(X_train[idx], w)[0]))
            # temp_sum += t.reshape(-1,1)

            # temp_sum += y_train[idx][0] * (X_train[idx]).reshape(-1, 1) / (1 + math.exp(y_train[idx] * np.dot(X_train[idx], w)[0]))

            temp_sum += X_train[idx] * (y_train[idx][0] - logistic_g(np.dot(X_train[idx], w)[0]))

        # objective function
        # w = (1 - learning_rate * reg_param) * w + learning_rate / batch_size * temp_sum

        # loss function
        w = learning_rate * n_train_samples / batch_size * temp_sum

        # print('w = ', np.floor(w.reshape(1, -1)))

        loss_train = threshold_Ein(X_train, y_train, w)
        losses_train.append(loss_train)

        loss_val = threshold_Ein(X_val, y_val, w)
        losses_val.append(loss_val)

        # print(f"epoch {epoch}: loss_train = {loss_train}; loss_val = {loss_val}")
        print("epoch {}: loss_train = [{:.2f}]; loss_val = [{:.2f}]".format(epoch, loss_train, loss_val))

    return w, losses_train, losses_val

def rnd_sample_indice(low, high, cnt):
    """ select {cnt} distinct integers (samples) from the range [low, high)
    """
    list = [x for x in range(low, high)]
    np.random.seed()
    np.random.shuffle(list)
    return list[:cnt]

# def rnd_sample_indice2(high, cnt):
#     return random.sample(range(0, high), cnt)

def logistic_g(Z):
    return 1 / (1 + math.exp(-Z))

def threshold_loss(X, y, w, threshold=0.5):
    n_samples = X.shape[0]
    y_predict = np.dot(X, w)
    for i in range(0, n_samples):
        if logistic_g(y_predict[i]) > threshold:
            y_predict[i] = +1
        else:
            y_predict[i] = -1

    # 这样的损失函数怪怪的
    return np.average(np.abs(y_predict - y))

def threshold_Ein(X, y, w, threshold=0.5):
    n_samples = X.shape[0]

    y_predict = np.dot(X, w)
    for i in range(0, n_samples):
        if logistic_g(y_predict[i]) > threshold:
            y_predict[i] = +1
        else:
            y_predict[i] = -1

    loss_sum = 0
    for i in range(0, n_samples):
        loss_sum += math.log(1 + np.exp(-y[i] * y_predict[i]))

    return loss_sum / n_samples


def loss(X, y, w):
    n_samples = X.shape[0]
    loss_sum = 0
    for i in range(0, n_samples):
        loss_sum += np.log(1 + np.exp(-y[i] * (np.dot(X[i], w))[0]))

    return loss_sum / n_samples


def run_SGD():
    global n_features
    X_train, y_train = preprocess(dataset_url=train_dataset_url, n_features=n_features)
    X_val, y_val = preprocess(dataset_url=val_dataset_url, n_features=n_features)
    w, losses_train, losses_val = log_reg_SGD(X_train, y_train, X_val, y_val, batch_size=512, max_epoch=200)

    plt.figure(figsize=(16,9))
    plt.plot(losses_train, "-", color="r", label="train loss")
    plt.plot(losses_val, "-", color='b', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('loss graph')
    plt.show()

if __name__ == "__main__":
    run_SGD()
