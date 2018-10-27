'''use both the closed-form solution and gradient descent method to solve a simple regression problem
dataset: housing_scale from LIBSVM
'''

# data_file = '../dataset/housing.txt'
data_file = '../dataset/housing_scale.txt'
n_features = 13

# --------------------------

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def preprocess():
    global n_features

    X, y = load_svmlight_file(data_file, n_features = n_features)

    X = X.toarray()
    # X = np.column_stack((np.ones((X.shape[0],1)), X))
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n_features += 1

    y = y.reshape(X.shape[0], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    return X_train, y_train, X_test, y_test

def closed_form(X_train, y_train, X_test, y_test):
    global n_features

    # init weight vector
    w = np.zeros(n_features)
    # w = np.random.random(n_features)
    # w = np.random.normal(size=(n_features))

    loss = least_square_loss(X_train, y_train, w)

    # X_train_T = np.transpose(X_train)
    X_train_T = X_train.T

    # t = np.dot(X_train_T, X_train)
    # t = np.mat(t).I
    # t = t.getA()

    w = np.dot(np.dot((np.mat(np.dot(X_train_T, X_train)).I).getA(), X_train_T), y_train)

    loss_train = least_square_loss(X_train, y_train, w)
    loss_val = least_square_loss(X_test, y_test, w)

    return w, loss, loss_train, loss_val

def GD(X_train, y_train, X_test, y_test, epoches=200, learning_rate=0.01, penalty_factor = 0.01):
    global n_features
    # init weight vector
    w = np.zeros((n_features, 1))
    # w = np.random.random(n_features)
    # w = np.random.normal(size=(n_features, 1))

    losses_train = []
    losses_test = []

    for epoch in range(0, epoches):
        # diff = np.dot(X_train, w) - y_train
        # G = penalty_factor * w + np.dot(X_train.transpose(), diff)  # calculate the gradient
        # G = -G
        # w += learning_rate * G  # update the parameters
        #
        # Y_predict = np.dot(X_train, w)  # predict under the train set
        # loss_train = np.average(np.abs(Y_predict - y_train))  # calculate the absolute differences
        # losses_train.append(loss_train)
        #
        # Y_predict = np.dot(X_test, w)  # predict under the validation set
        # loss_test = np.average(np.abs(Y_predict - y_test))  # calculate the absolute differences
        # losses_test.append(loss_test)
        #
        # print(f"at epoch [{epoch}]: loss_train = [{loss_train}]; loss_val = [{loss_test}]")

        d = -(penalty_factor * w - np.dot(X_train.T, y_train) + np.dot(np.dot(X_train.T, X_train), w)) # 没有正则项
        w += learning_rate * d

        print("at epoch [{:3d}]: loss_train = [{:.2f}; loss_val = [{:.2f}]".format(epoch, least_square_loss(X_train, y_train, w), least_square_loss(X_test, y_test, w)))

    loss_train = least_square_loss(X_train, y_train, w)
    loss_val = least_square_loss(X_test, y_test, w)

    return w, loss_train, loss_val

def least_square_loss(X, y, w):
    loss_sum = 0
    n_samples = X.shape[0]
    for i in range(0, n_samples):
        interval = (np.dot(X[i], w) - y[i])[0]
        loss_sum += interval * interval
    return loss_sum / n_samples


def run_closed_form():
    w, loss, loss_train, loss_val = closed_form(*preprocess())
    print("w = ", w)
    print('loss = {:.2f}'.format(loss))
    print('loss = {:.2f}'.format(loss_train))
    print('loss = {:.2f}'.format(loss_val))

def run_GD():
    X_train, y_train, X_test, y_test = preprocess()
    epoches = 200
    learning_rate = 0.001
    penalty_factor = 0.5

    w, loss_train, loss_val = GD(X_train, y_train, X_test, y_test, epoches = epoches, learning_rate = learning_rate, penalty_factor=penalty_factor)
    print("w = ", w)
    print('loss = {:.2f}'.format(loss_train))
    print('loss = {:.2f}'.format(loss_val))


if __name__ == "__main__":
    # run_closed_form()
    run_GD()

