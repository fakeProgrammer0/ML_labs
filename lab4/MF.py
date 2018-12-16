import numpy as np
import math
import random
import matplotlib.pyplot as plt

dataset_dir = __file__ + '/../ml-100k/'
train_set_path = dataset_dir + 'u1.base'
test_set_path = dataset_dir + 'u1.test'

n_users = 943
n_items = 1682


def load_dataset(dataset_path):
    '''
    Load the dataset and return a rating matrix.
    
    Returns
    -------
    R : ndarray
        The observed rating matrix with null values filled 0

    '''
    R = np.zeros((n_users, n_items))
    dataset_matrix = np.loadtxt(dataset_path, dtype=np.int32)
    for u, i, r, _ in dataset_matrix:
        R[u - 1, i - 1] = r
    return R


R_train = load_dataset(train_set_path)
R_test = load_dataset(test_set_path)


def MF_RMSE(R, P, Q):
    '''
    Evaluate the RMSE (Root Mean Square Error) of groundtruth rating matrix 
    and the matrix factorization model.

    Parameters
    ----------
    R : ndarray
        The groundtruth rating matrix, an ndarray of shape(n_users, n_items), 
        where R[i, j] denotes the rating user i marks movie j.
    P : ndarray
        The User-Feature matrix, an ndarray of shape(K, n_users), 
        where P[k, i] denotes user i's preferred score for latent feature k.
    Q : ndarray
        The Feature-Movie matrix, an ndarray of shape(K, n_items), 
        where Q[k, j] denotes movie j's score for latent feature k.

    Returns
    -------
    RMSE : float
        The RMSE between R and np.dot(P.T, Q)
    '''
    R_hat = np.dot(P.T, Q)
    R_hat[R == 0] = 0
    A_01 = R != 0
    # divide the number of observed ratings
    return math.sqrt(np.sum((R - R_hat)**2) / np.sum(A_01))  


def MF_MAE(R, P, Q):
    '''
    Evaluate the MAE (Mean Absolute Error) of groundtruth rating matrix 
    and the matrix factorization model.

    Parameters
    ----------
    R : ndarray
        The groundtruth rating matrix, an ndarray of shape(n_users, n_items), 
        where R[i, j] denotes the rating user i marks movie j.
    P : ndarray
        The User-Feature matrix, an ndarray of shape(K, n_users), 
        where P[k, i] denotes user i's preferred score for latent feature k.
    Q : ndarray
        The Feature-Movie matrix, an ndarray of shape(K, n_items), 
        where Q[k, j] denotes movie j's score for latent feature k.

    Returns
    -------
    MAE : float
        The MAE between R and np.dot(P.T, Q)
    '''
    R_hat = np.dot(P.T, Q)
    R_hat[R == 0] = 0
    A_01 = R != 0
    return np.sum(np.abs(R - R_hat)) / np.sum(A_01)


def MF_SGD_fit(R_train,
               R_test,
               K,
               learning_rate,
               max_epoch,
               reg_lambda_p,
               reg_lambda_q,
               min_loss_threshold=0.1,
               loss_estimate=MF_RMSE,
               epoch_cnt_per_loss_estimate=1000):
    """
    Fit a rating matrix and optimize the matrix factorization model using SGD method.

    Parameters
    ----------
    R_train : ndarray 
        The groundtruth rating matrix used for training, in shape (n_users, n_items).
    R_test : ndarray 
        The groundtruth rating matrix for testing, in shape (n_users, n_items).
    K : int
        The number of latent features.
    learning_rate : float
        The hyper-parameter to control the velocity of gradient descent process, 
        also called step_size.
    max_epoch : int
        The number of training epoches.
    reg_lambda_p, reg_lambda_q : float
        The regularization parameters of the model cost term.
    min_cost_threshold : float
        When the training cost reaches or is lower than the thresold, training will stop.
    loss_estimate :  callable
        A custom loss evaluation function with following signature (R, P, Q) 
        returns the loss of the matrix factorization model. 
        The default setting is using RMSE.
    epoch_cnt_per_loss_estimate : int
        Loss will be estimated at every epoch count.

    Returns
    -------
    R_pred : ndarray
        The predicted rating matrix.

    losses_dict : dict
        A dict containing the model's losses on training and testing dataset 
        during the training procedure.

    """

    n_users, n_items = R_train.shape

    P, Q = np.random.rand(K, n_users), np.random.rand(K, n_items)

    losses_train = []
    losses_test = []

    # acquire observed rating (u, i) pairs to support random selecting efficently
    observed_rating_ui_pairs = []
    for u in range(n_users):
        for i in range(n_items):
            if R_train[u, i]:
                observed_rating_ui_pairs.append((u, i))

    random.shuffle(observed_rating_ui_pairs)

    for epoch in range(max_epoch):

        u, i = random.choice(observed_rating_ui_pairs)

        e_ui = R_train[u, i] - P[:, u] @ Q[:, i]
        # P[:, u] += learning_rate * (2*e_ui*Q[:, i] - reg_lambda_p*P[:, u])
        # Q[:, i] += learning_rate * (2*e_ui*P[:, u] - reg_lambda_q*Q[:, i])

        P[:, u], Q[:, i] = P[:, u] + learning_rate * (2*e_ui*Q[:, i] - reg_lambda_p*P[:, u]), \
                            Q[:, i] + learning_rate * (2*e_ui*P[:, u] - reg_lambda_q*Q[:, i])

        if epoch % epoch_cnt_per_loss_estimate == 0:
            curr_train_loss = loss_estimate(R_train, P, Q)
            losses_train.append(curr_train_loss)

            curr_val_loss = loss_estimate(R_test, P, Q)
            losses_test.append(curr_val_loss)

        if curr_train_loss < min_loss_threshold:
            break

    R_pred = P.T @ Q

    losses_dict = {
        'losses_train': losses_train,
        'losses_test': losses_test,
    }

    return R_pred, losses_dict


def MF_ALS_fit(R_train,
               R_test,
               K,
               reg_lambda,
               max_epoch,
               min_RMSE_threshold=0.1,
               loss_estimate=MF_RMSE):
    """
    Fit a rating matrix and optimize the matrix factorization model using ALS method.

    Parameters
    ----------
    R_train : ndarray 
        The groundtruth rating matrix used for training, in shape (n_users, n_items).
    R_test : ndarray 
        The groundtruth rating matrix for testing, in shape (n_users, n_items).
    K : int
        The number of latent features.
    reg_lambda : float
        The regularization parameter lambda of the model cost term.
    max_epoch : int
        The number of training epoches.
    min_cost_threshold : float
        When the training cost reaches or is lower than the thresold, training will stop.
    loss_estimate :  callable
        A custom loss evaluation function with following signature (R, P, Q) 
        returns the loss of the matrix factorization model. 
        The default setting is using RMSE.

    Returns
    -------
    R_pred : ndarray
        The predicted rating matrix.

    losses_dict : dict
        A dict containing the model's losses on training and testing dataset 
        during the training procedure.

    """
    assert isinstance(R_train, np.ndarray)
    n_users, n_items = R_train.shape

    # 留意矩阵的维度
    P, Q = np.random.random((K, n_users)), np.random.random((K, n_items))

    N_U = np.sum(R_train != 0, axis=1)  # 用户评分的次数 shape: (n_users)

    N_M = np.sum(R_train != 0, axis=0)  # 电影被评分的次数 shape: (n_items)
    # 用一部电影的平均评分作为该电影的第0个隐含特征的分数
    Q[0, :] = np.sum(R_train, axis=0) / N_M  # shape: (2) <= shape: (2)
    # 有些电影在该数据集中没有被打分，相除后平均分数为无穷大
    for i in range(n_items):
        # if M[0, i] == np.nan:
        if not (0 <= Q[0, i] <= 5):
            Q[0, i] = 0

    losses_train, losses_test = [], []

    for epoch in range(max_epoch):

        # 必须把M_Ui, U_Mj这些小型矩阵抽解出来，而不是在庞大的原始矩阵上进行补0操作
        # 不然后面矩阵求逆会很麻烦，效率会很低，甚至因为0项太多，矩阵是不可逆的

        for i in range(n_users):
            M_Ui = None  # 把U[i]评价过的电影的特征列都挑选出来，组成一个(K * N_U[i])的小型矩阵
            R_Ui = None  # 把M_Ui对应的评分都挑选出来，组成一个(1 * N_U[i])的行向量
            for j in range(n_items):
                if R_train[i, j]:
                    if M_Ui is not None:
                        M_Ui = np.hstack((M_Ui, Q[:, j:j + 1]))
                        R_Ui = np.hstack((R_Ui, R_train[i:i + 1, j:j + 1]))
                    else:
                        M_Ui = Q[:, j:j + 1]
                        R_Ui = R_train[i:i + 1, j:j + 1]

            # 有些用户在该数据集中没有评价任何电影
            if M_Ui is None:
                continue

            Ai = M_Ui.dot(M_Ui.T) + reg_lambda * N_U[i] * np.eye(K)
            Vi = M_Ui.dot(R_Ui.T)

            P[:, i:i + 1] = np.dot(np.matrix(Ai).I.getA(), Vi)

        for j in range(n_items):
            U_Mj = None  # 把评价过电影M[j]的用户的喜好特征行挑选出来，组成一个(K * N_M[i])的小型矩阵
            R_Mj = None  # 把U_Mj对应的评分挑选出来 -- 一个(N_M[i] * 1)的列向量
            for i in range(n_users):
                if R_train[i, j]:
                    if U_Mj is not None:
                        U_Mj = np.hstack((U_Mj, P[:, i:i + 1]))
                        R_Mj = np.vstack((R_Mj, R_train[i:i + 1, j:j + 1]))
                    else:
                        U_Mj = P[:, i:i + 1]
                        R_Mj = R_train[i:i + 1, j:j + 1]

            # 有些电影在该数据集中没有被任何用户评价
            if U_Mj is None:
                continue

            Aj = np.dot(U_Mj, U_Mj.T) + reg_lambda * N_M[j] * np.eye(K)
            Vj = U_Mj.dot(R_Mj)

            Q[:, j:j + 1] = np.dot(np.matrix(Aj).I.getA(), Vj)

        curr_loss_train = loss_estimate(R_train, P, Q)
        curr_loss_test = loss_estimate(R_test, P, Q)
        losses_train.append(curr_loss_train)
        losses_test.append(curr_loss_test)

        if curr_loss_train <= min_RMSE_threshold:
            break

    R_pred = np.dot(P.T, Q)

    losses_dict = {
        'losses_train': losses_train,
        'losses_test': losses_test,
    }

    return R_pred, losses_dict


def plot_losses_graph(losses_dict,
                      title="loss graph",
                      xlabel="epoch",
                      ylabel='loss'):
    '''
    A helper function used to draw the losses graph.

    Parameters
    ----------
    losses_dict : dict
        A dict contains losses information which are in the form (losses_label, losses_data)
            losses_label : string
                A label indicates the information about the loss.
            losses_data : list
                A list consists of loss data.

    '''
    colors = ['r', 'b', 'k', 'g', 'c', 'm', 'y']

    plt.figure(figsize=(16, 9))
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    for i, losses_label in enumerate(losses_dict):
        losses_data = losses_dict.get(losses_label)
        plt.plot(
            losses_data,
            '-',
            color=colors[i % len(colors)],
            label=losses_label)

    plt.legend()
    plt.show()


def train_SGD():

    param_dict = {
        'K': 30,
        'reg_lambda_p': 0.1,
        'reg_lambda_q': 0.1,
        'learning_rate': 0.001,
        'max_epoch': 100000,
        'loss_estimate': MF_RMSE,
        'epoch_cnt_per_loss_estimate': 1000
    }

    R_pred, losses_dict = MF_SGD_fit(R_train.copy(), R_test.copy(), **param_dict)
    plot_losses_graph(
        losses_dict,
        title='loss during SGD',
        xlabel='%d epoches' % param_dict['epoch_cnt_per_loss_estimate'],
        ylabel='RMSE')


def train_ALS():
    param_dict = {
        'K': 30,
        'reg_lambda': 0.1,
        'max_epoch': 20,
        'loss_estimate': MF_RMSE
    }

    R_pred, losses_dict = MF_ALS_fit(R_train.copy(), R_test.copy(), **param_dict)
    plot_losses_graph(
        losses_dict,
        title='loss during ALS\nK=%d, reg_lambda=%.6f' %
        (param_dict['K'], param_dict['reg_lambda']),
        ylabel='RMSE')


def estimate_K():
    '''
    '''

    max_epoch = 30
    tuned_params = [
        {
            'K' : 5,
            'reg_lambda' : 0.01
        },
        {
            'K' : 20,
            'reg_lambda' : 0.1
        },
        {
            'K' : 50,
            'reg_lambda' : 0.01
        },
        {
            'K' : 100,
            'reg_lambda' : 0.01
        },
    ]

    losses_train_dict, losses_test_dict = {}, {}
    
    for param_dict in tuned_params:
        R_pred, losses_dict = MF_ALS_fit(R_train.copy(), R_test.copy(), max_epoch=max_epoch, **param_dict)
        losses_train_dict['K=%d,reg_lambda=%d' % (param_dict['K'], param_dict['reg_lambda'])] = losses_dict['losses_train']
        losses_test_dict['K=%d,reg_lambda=%d' % (param_dict['K'], param_dict['reg_lambda'])] = losses_dict['losses_test']

    plot_losses_graph(losses_train_dict, title='ALS losses train vary with different K')
    plot_losses_graph(losses_test_dict, title='ALS losses test vary with different K')


def estimate_reg_lambda():
    '''
    '''
    K = 30
    max_epoch = 15
    turn_params = [0.005, 0.02, 0.08, 0.1, 1.0, 5.0]

    losses_train_dict, losses_test_dict = {}, {}

    for reg_lambda in turn_params:
        R_pred, losses_dict = MF_ALS_fit(R_train.copy(), R_test.copy(), K, reg_lambda, max_epoch)
        losses_train_dict['reg_lambda=%d' % reg_lambda] = losses_dict['losses_train']
        losses_test_dict['reg_lambda=%d' % reg_lambda] = losses_dict['losses_test']

    plot_losses_graph(losses_train_dict, title=f'ALS fixing K={K}\nlosses train vary with different reg_lambda')
    plot_losses_graph(losses_test_dict, title=f'ALS fixing K={K}\nlosses test vary with different reg_lambda')

if __name__ == '__main__':
    # train_ALS()
    # train_SGD()
    estimate_K()
    pass
