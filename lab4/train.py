# -*- coding: utf-8 -*-
'''Training a movie recommender system based on the ml-100k dataset

'''

# ------ Hyperparameters to be tuned -------


# ---------------

import copy
import math
import random
import numpy as np
from string import Template
import matplotlib.pyplot  as plt

# import sys
# sys.path.append(".")

# from . import matrix_factorization
# from lab4 import matrix_factorization
from lab4.matrix_factorization import MF_SGD
from lab4.matrix_factorization import MF_ALS_Model

dataset_path_temp = Template("./ml-100k/{dataset_filename}")

n_users = 943
n_items = 1682

# --------------------


def load_dataset(dataset_path):

    R = np.zeros((n_users, n_items))
    A_01 = np.zeros((n_users, n_items))

    with open(dataset_path, 'r', encoding='UTF-8') as data_fp:
        for line in data_fp:
            line = line.strip()
            if line:  # skip empty lines if any exists
                line = line.split("\t")
                user_id, item_id, rating = int(line[0]) - 1, int(line[1]) - 1, int(line[2])
                R[user_id, item_id] = rating
                A_01[user_id, item_id] = 1

    return R, A_01


# def sample_err(R_row, P_row, Q, reg_lambda):
#     assert R_row is np.array and R_row.shape[0] == 1 \
#         and P_row is np.array \
#         and Q is np.array
#
#     err_sum = 0
#     for j in range(R_row.shape):
#         if R_row[j] != 0:
#             temp_diff = R_row[j] - np.dot(P_row, Q[j])[0]
#             err_sum += temp_diff * temp_diff + reg_lambda * (P_row.dot(P_row)[0] + np.dot(Q[j], Q[j])[0])
#
#     return err_sum


def train_MF_SGD_Model():
    K = 40
    n_folds = 5

    base_dataset_temp = Template("./ml-100k/u${i}.base")
    test_dataset_temp = Template("./ml-100k/u${i}.test")

    for i in range(1, n_folds+1):
        R_train, A_01_train = load_dataset(base_dataset_temp.substitute(i=i))
        R_test, A_01_test = load_dataset(test_dataset_temp.substitute(i=i))

        SGD_model = MF_SGD()

        R_train_pred, train_losses, val_losses = SGD_model.losses_estimate(R_train, R_test, K, learning_rate=0.001, max_epoch=2000, reg_lambda=0.5)
        plot_losses_graph(train_losses, val_losses, 'loss estimate of fold %d' % i)


def train_MF_ALS_Model():
    K = 40
    n_folds = 5

    base_dataset_temp = Template("./ml-100k/u${i}.base")
    test_dataset_temp = Template("./ml-100k/u${i}.test")

    for i in range(1, n_folds + 1):
        R_train, A_01_train = load_dataset(base_dataset_temp.substitute(i=i))
        R_test, A_01_test = load_dataset(test_dataset_temp.substitute(i=i))

        ALS_Model = MF_ALS_Model()

        # ALS_Model.fit(R_train, K, max_epoch=200, reg_lambda=0.5)
        R_train_pred, train_losses, val_losses = ALS_Model.cost_estimate(R_train, R_test, K, max_epoch=10, reg_lambda=0.5)
        plot_losses_graph(train_losses, val_losses, 'loss estimate of fold %d' % i)


    pass

def plot_losses_graph(train_losses, val_losses, title="loss graph"):
    plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_losses, '-', color='r', label='train_losses')
    plt.plot(val_losses, '--', color='b', label='val_losses')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # train_MF_SGD_Model()
    train_MF_ALS_Model()
    pass





