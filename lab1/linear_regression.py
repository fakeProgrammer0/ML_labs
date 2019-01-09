'''simple linear regression
methodology: use both the closed-form solution and gradient descent method
dataset: housing_scale from LIBSVM Data
'''

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('./')
from tools import plot_losses_graph
from tools import download_dataset
from tools import execute_procedure

# dataset_url = '''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing'''
dataset_url = '''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale'''
dataset_file_path = download_dataset(dataset_url)
n_features = 13


def preprocess(dataset_file_path, n_features, test_size=0.25):
    X, y = load_svmlight_file(dataset_file_path, n_features=n_features)

    y = y.reshape(-1, 1)
    X = X.toarray()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size)
    return X_train, y_train, X_val, y_val


def linear_reg_closed_form(X_train, y_train, X_val, y_val):
    '''Use the closed-form solution to optimize simple linear regression.
    Attention: This function may not work because the inverse of a given 
            matrix may not exist.

    Parameters
    ----------
    X_train: array-like of shape = (n_train_samples, n_features)
        Samples of training set.

    y_train: array-like of shape = (n_train_samples, 1)
        Groundtruth labels of training set.

    X_val: array-like of shape = (n_val_samples, n_features)
        Samples of validation set.

    y_val: array-like of shape = (n_val_samples, 1)
        Groundtruth labels of validation set.

    Returns
    -------
    w: array-like of shape = (n_features, 1)
        The weight vector.

    b: int
        The bias of linear regression.

    losses_dict: dict
        A dict containing losses evaluated before and after training

    '''
    n_features = X_train.shape[1]

    # make all X[i, 0] = 1
    X_train = np.hstack((np.ones(y_train.shape), X_train))
    X_val = np.hstack((np.ones(y_val.shape), X_val))

    # init weight vector
    w = np.zeros((n_features + 1, 1))  # zero based weight vector
    # w = np.random.random((n_features+1, 1)) # initialize with small random values
    # w = np.random.normal(1, 1, size=(n_features+1, 1))

    losses_dict = {}
    losses_dict['losses_train_origin'] = mean_squared_error(
        y_true=y_train, y_pred=np.dot(X_train, w))
    losses_dict['losses_val_origin'] = mean_squared_error(
        y_true=y_val, y_pred=np.dot(X_val, w))

    # use closed-form solution to update w
    # @ operation equals to np.dot
    try:
        w = np.mat(X_train.T @ X_train).I.getA() @ X_train.T @ y_train
    except Exception as ex:
        print('The inverse of the matrix X_train.T @ X_train doesn\'t exist.')
        print(ex)

    losses_dict['losses_train'] = mean_squared_error(y_train, np.dot(
        X_train, w))
    losses_dict['losses_val'] = mean_squared_error(y_val, np.dot(X_val, w))

    w, b = w[1:, ], w[0, 0]

    return w, b, losses_dict


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
    n_features = X_train.shape[1]

    # make all X[i, 0] = 1
    X_train = np.hstack((np.ones(y_train.shape), X_train))
    X_val = np.hstack((np.ones(y_val.shape), X_val))

    # init weight vector
    w = np.zeros((n_features + 1, 1))  # zero based weight vector
    # w = np.random.random((n_features+1, 1)) # initialize with small random values
    # w = np.random.normal(1, 1, size=(n_features+1, 1))

    losses_train, losses_val = [], []

    for epoch in range(0, max_epoch):
        d = -penalty_factor * w + X_train.T @ (y_train - X_train @ w)
        w += learning_rate * d

        # update learning rate if necessary
        # learning_rate /= (epoch + 1) # emmm...no so good
        # learning_rate /= 1 + learning_rate * penalty_factor * (epoch + 1)

        loss_train = mean_squared_error(
            y_true=y_train, y_pred=np.dot(X_train, w))
        loss_val = mean_squared_error(y_true=y_val, y_pred=np.dot(X_val, w))
        losses_train.append(loss_train)
        losses_val.append(loss_val)

    w, b = w[1:, ], w[0, 0]

    losses_dict = {'losses_train': losses_train, 'losses_val': losses_val}

    return w, b, losses_dict


def run_closed_form():
    global dataset_file_path, n_features
    w, b, losses_dict = linear_reg_closed_form(
        *preprocess(dataset_file_path, n_features))

    print(
        'Mean Square Errors for linear regression using closed-form solution')
    for losses_label in losses_dict:
        losses_value = losses_dict[losses_label]
        print('%10s = %.6f' % (losses_label, losses_value))


def run_GD():
    params_dict = {
        'max_epoch': 200,
        'learning_rate': 0.0005,
        'penalty_factor': 0.5
    }

    X_train, y_train, X_val, y_val = preprocess(dataset_file_path, n_features)
    w, b, losses_dict = linear_reg_GD(X_train, y_train, X_val, y_val,
                                      **params_dict)

    plot_losses_graph(
        losses_dict,
        'Mean Square Errors for linear regression using GD',
        params_dict=params_dict)


def estimate_GD_learning_rate():
    params_dict = {
        'max_epoch' : 200,
        "penalty_factor" : 0.5
    }

    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    losses_train_dict, losses_test_dict = {}, {}

    for learning_rate in learning_rates:
        X_train, y_train, X_val, y_val = preprocess(dataset_file_path,
                                                    n_features)
        w, b, losses_dict = linear_reg_GD(
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=learning_rate,
            **params_dict)
        losses_train_dict[f'learning_rate = {learning_rate}'] = losses_dict[
            'losses_train']
        losses_test_dict[f'learning_rate = {learning_rate}'] = losses_dict[
            'losses_val']

    plot_losses_graph(
        losses_train_dict,
        title='losses_train vary with different learning_rate',
        ylabel='Mean Square Error',
        params_dict=params_dict,
        params_notation_pos_width_perc=0.85,
        params_notation_pos_height_perc=0.7)
    plot_losses_graph(
        losses_test_dict,
        title='losses_train vary with different learning_rate',
        ylabel='Mean Square Error',
        params_dict=params_dict,
        params_notation_pos_width_perc=0.85,
        params_notation_pos_height_perc=0.7)


if __name__ == "__main__":
    execute_procedure(run_closed_form, 'run_closed_form')
    execute_procedure(run_GD, 'run_GD')
    execute_procedure(estimate_GD_learning_rate, 'estimate_GD_learning_rate')
    print('Done!')
