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


def log_reg_MSGD_MLE(X_train, y_train, X_val, y_val, batch_size=100, max_epoch=200, learning_rate=0.001, reg_param=0.3):
    '''logistic regression using the mini-batch stochastic gradient descent method with maximum likelihood loss function

    :param y_train: train_labels in the column shape, where y_train[i] is either 0 or 1
    :return w: the weighted vector
    '''

    def neg_log_likelihood(X, y, w):
        '''calculate the loss of logistic regression using the maximum likelihood method
        :param X:
        :param y: labels, where y[i] is either 0 or 1
        :param w: the weighted vector, in a row shape
        :return:
        '''

        # TODO: check dimension equality constraints

        n_samples = X.shape[0]
        loss_sum = 0
        for i in range(0, n_samples):
            # loss_sum += y[i][0] * np.dot(X[i], w) - math.log(1 + math.exp(np.dot(X[i], w))) # overflowerror
            loss_sum += (y[i][0] - 1) * np.dot(X[i], w) - math.log(1 + math.exp(-np.dot(X[i], w)))
        return -loss_sum

    # map y[i] from {-1.0, +1.0} into {0.0, 1.0}
    if y_train.min() == np.float64(-1.0) or y_val.min() == np.float64(-1.0):
        y_train = (y_train + np.ones(y_train.shape)) / 2
        y_val = (y_val + np.ones(y_val.shape)) / 2

    global n_features
    # init weight vectors
    # w = np.zeros((n_features + 1, 1))

    # for calculation convenience, w is represented as a row vector
    w = np.zeros(n_features + 1)

    n_train_samples = X_train.shape[0]
    if n_train_samples < batch_size:
        batch_size = n_train_samples

    neg_log_LE_train = []
    neg_log_LE_val = []

    for epoch in range(0, max_epoch):

        temp_sum = np.zeros(n_features + 1)
        batch_indice = random.sample(range(0, n_train_samples), batch_size)

        for idx in batch_indice:
            temp_sum += X_train[idx] * (y_train[idx][0] - logistic_g(np.dot(X_train[idx], w)))

        # update w using gradient
        w -= learning_rate * n_train_samples / batch_size * temp_sum

        # print('w = ', np.floor(w.reshape(1, -1)))

        loss_train = neg_log_likelihood(X_train, y_train, w)
        neg_log_LE_train.append(loss_train)

        loss_val = neg_log_likelihood(X_val, y_val, w)
        neg_log_LE_val.append(loss_val)

        # print("epoch {}: loss_train = [{:.2f}]; loss_val = [{:.2f}]".format(epoch, loss_train, loss_val))

    w = w.reshape(-1, 1)
    return w, neg_log_LE_train, neg_log_LE_val

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
    w, MLEs_train, MLEs_val = log_reg_MSGD_MLE(X_train, y_train, X_val, y_val, batch_size=512, max_epoch=200)

    plt.figure(figsize=(16,9))
    plt.plot(MLEs_train, "-", color="r", label="train MLE")
    plt.plot(MLEs_val, "-", color='b', label='val MLE')
    plt.xlabel('epoch')
    plt.ylabel('MLE')
    plt.legend()
    plt.title('MLE graph')
    plt.show()

if __name__ == "__main__":
    run_SGD()
