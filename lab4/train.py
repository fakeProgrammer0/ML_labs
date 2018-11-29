# -*- coding: utf-8 -*-
'''Training a movie recommender system based on the ml-100k dataset
Using SGD or ALS method for matrix factorization
'''

# ------ Hyperparameters to be tuned -------


# ---------------

import copy
import math
import random
import numpy as np
from string import Template
import matplotlib.pyplot  as plt

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

        # reg_lambda = 0.08
        reg_lambda = 0.1

        # ALS_Model.fit(R_train, K, max_epoch=5, reg_lambda=0.5)

        R_train_pred, losses_dict = ALS_Model.cost_estimate(R_train, R_test, K, max_epoch=20, reg_lambda=reg_lambda)
        plot_losses_graph(losses_dict, 'Losses of ALS Model during training\nreg_lamda = %.6f' % reg_lambda)


    pass

def plot_losses_graph(losses_dict, title="loss graph"):
    colors = ['r', 'b', 'k', 'g', 'c', 'm', 'y']

    plt.figure(figsize=(16, 9))
    plt.title(title, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    for i, losses_label in enumerate(losses_dict):
        losses_data = losses_dict.get(losses_label)
        plt.plot(losses_data, '-', color=colors[i % len(colors)], label=losses_label)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # train_MF_SGD_Model()
    train_MF_ALS_Model()
    pass





