'''use both the closed-form solution and gradient descent method to solve a simple regression problem
dataset: housing_scale from LIBSVM Data
'''

# data_file_url = '../dataset/housing.txt'
dataset_file_url = '../dataset/housing_scale.txt'
n_features = 13

# --------------------------

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def preprocess(dataset_file_url, n_features, test_size=0.25):
    X, y = load_svmlight_file(dataset_file_url, n_features=n_features)

    y = y.reshape(-1, 1)
    X = X.toarray()
    X = np.hstack((np.ones(y.shape), X))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_val, y_val


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


def linear_reg_GD(X_train, y_train, X_val, y_val, max_epoch=200, learning_rate=0.01, penalty_factor=0.5):
    '''Use the gradient descent method to solve simple linear regression.
    :param X_train: train data, a (n_samples, n_features + 1) ndarray, where the 1st column are all ones, ie.numpy.ones(n_samples)
    :param y_train: labels, a (n_samples, 1) ndarray
    :param X_val: validation data
    :param y_val: validation labels
    :param max_epoch: the max epoch for training
    :param learning_rate: the hyper parameter to control the velocity of gradient descent process, also called step_size
    :param penalty_factor: the L2 regular term factor for the objective function

    :return w: the weight vector, a (n_features + 1, 1) ndarray
    :return losses_train: the mean square loss of the training set during each epoch
    :return losses_val: the mean square loss of the validation set during each epoch
    '''

    n_features = X_train.shape[1]
    # init weight vector
    w = np.zeros((n_features, 1))
    # w = np.random.random(n_features)
    # w = np.random.normal(1, 1, size=(n_features, 1))

    losses_train = []
    losses_val = []

    for epoch in range(0, max_epoch):
        d = -penalty_factor * w + np.dot(X_train.T, (y_train - np.dot(X_train, w)))
        w += learning_rate * d

        # update learning rate if necessary
        # learning_rate /= (epoch + 1) # emmm...no so good
        # learning_rate /= 1 + learning_rate * penalty_factor * (epoch + 1)

        loss_train = mean_squared_error(y_true=y_train, y_pred=np.dot(X_train, w))
        loss_val = mean_squared_error(y_true=y_val, y_pred=np.dot(X_val, w))
        losses_train.append(loss_train)
        losses_val.append(loss_val)

        print("at epoch [{:4d}]: loss_train = [{:.6f}; loss_val = [{:.6f}]".format(epoch, loss_train, loss_val))

    return w, losses_train, losses_val


def run_closed_form():
    global dataset_file_url, n_features
    w, loss0, loss1, loss_train, loss_val = linear_reg_closed_form(*preprocess(dataset_file_url, n_features))
    print('closed-form solution for linear regression')
    print('%10s = %.6f' % ('loss0', loss0))
    print('%10s = %.6f' % ('loss1', loss1))
    print('%10s = %.6f' % ('loss_train', loss_train))
    print('%10s = %.6f' % ('loss_val', loss_val))
    # print("the weight vector w : \n", w)


import matplotlib.pyplot as plt

def run_GD():
    max_epoch = 200
    learning_rate = 0.0005
    penalty_factor = 0.5

    X_train, y_train, X_val, y_val = preprocess(dataset_file_url, n_features)
    w, losses_train, losses_val = linear_reg_GD(X_train, y_train, X_val, y_val, max_epoch=max_epoch, learning_rate=learning_rate, penalty_factor=penalty_factor)

    plt.figure(figsize=(16, 9))
    plt.plot(losses_train, '--', color='k', label='training loss')
    plt.plot(losses_val, '-', color='r', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('mean square loss')
    plt.title(
        f'loss graph of linear regression using GD\nlearning_rate: {learning_rate}\npenlaty_factor: {penalty_factor}')
    plt.legend()
    plt.show()

def estimate_learning_rate_GD():
    max_epoch = 50
    penalty_factor = 10

    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    colors = ['c', 'g', 'm', 'k', 'r']

    plt.figure(figsize=(16, 9))
    for learning_rate, color in zip(learning_rates, colors):
        X_train, y_train, X_val, y_val = preprocess(dataset_file_url, n_features)
        w, losses_train, losses_val = linear_reg_GD(X_train, y_train, X_val, y_val, max_epoch=max_epoch, learning_rate=learning_rate, penalty_factor=penalty_factor)

        plt.plot(losses_val, '-', color=color, label='learing_rate = %.e' % learning_rate)

    plt.xlabel('epoch')
    plt.ylabel('mean square loss')
    plt.title(f'estimate learning rates\npenlaty_factor: {penalty_factor}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # run_closed_form()
    run_GD()
    # estimate_learning_rate_GD()
