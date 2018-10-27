'''use both the closed-form solution and gradient descent method to solve a simple regression problem
dataset: housing_scale from LIBSVM
'''

# data_file_url = '../dataset/housing.txt'
data_file_url = '../dataset/housing_scale.txt'
n_features = 13

# --------------------------

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def preprocess(data_file_url, n_features, test_size=0.25):
    X, y = load_svmlight_file(data_file_url, n_features = n_features)

    y = y.reshape(-1, 1)
    X = X.toarray()
    X = np.hstack((np.ones(y.shape), X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

from sklearn.metrics import mean_squared_error
def linear_reg_closed_form(X_train, y_train, X_val, y_val):
    '''Use the closed-form solution to solve simple linear regression.
    Attention: This function may not work because the inverse of a given matrix may not exist.
    :param X_train: train data, a (n_samples, n_features + 1) ndarray, where the 1st column are all ones, ie.numpy.ones(n_samples)
    :param y_train: labels, a (n_samples, 1) ndarray
    :param X_val: validation data
    :param y_val: validation labels
    :return w: the weight vector, a (n_features + 1, 1) ndarray
    '''

    n_features = X_train.shape[1]

    # init weight vector
    w = np.zeros((n_features, 1))
    # w = np.random.random((n_features, 1))
    # w = np.random.normal(1, 1, size=(n_features, 1))

    loss0 = mean_squared_error(y_true=y_train, y_pred=np.dot(X_train, w))

    w = np.dot(np.dot((np.mat(np.dot(X_train.T, X_train)).I).getA(), X_train.T), y_train)

    loss1 = mean_squared_error(y_true=y_train, y_pred=np.dot(X_train, w))
    loss_train = mean_squared_error(y_train, np.dot(X_train, w))
    loss_val = mean_squared_error(y_val, np.dot(X_val, w))

    return w, loss0, loss1, loss_train, loss_val

def linear_reg_closed_form2(X_train, y_train, X_test, y_test):

    n_features = X_train.shape[1]

    # init weight vector
    # for calculation convenience, w is represented as a row vector
    w = np.zeros(n_features)
    # w = np.random.random(n_features)
    # w = np.random.normal(1, 1, size=(n_features))

    loss = least_square_loss(X_train, y_train, w)

    w = np.dot(np.dot((np.mat(np.dot(X_train.T, X_train)).I).getA(), X_train.T), y_train)[:, 0]

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
    '''a helper method calculating the least square loss
    :param X: the data ndarray, in shape (n_samples * n_features)
    :param y: the labels, a column vector
    :param w: a column weight vector, a (n_features, 1) ndarray
    :return: the least square loss
    '''
    loss_sum = 0
    n_samples = X.shape[0]
    for i in range(0, n_samples):
        interval = np.dot(X[i], w) - y[i][0]
        loss_sum += interval * interval
    return loss_sum / n_samples

def run_closed_form():
    global data_file_url, n_features
    w, loss0, loss1, loss_train, loss_val = linear_reg_closed_form(*preprocess(data_file_url, n_features))
    print('closed-form solution for linear regression')
    print('%10s = %.6f' % ('loss0', loss0))
    print('%10s = %.6f' % ('loss1', loss1))
    print('%10s = %.6f' % ('loss_train',loss_train))
    print('%10s = %.6f' % ('loss_val', loss_val))
    # print("the weight vector w : \n", w)

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
    run_closed_form()
    # run_GD()

