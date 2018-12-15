import math
import random
import numpy as np


class MF_SGD:
    def __init__(self):
        pass

    def fit(self, R, K, learning_rate, max_epoch, reg_lambda):
        assert R is np.ndarray
        n_users, n_items = R.shape

        P, Q = np.zeros((n_users, K)), np.zeros((n_items, K))

        for epoch in range(max_epoch):
            i = random.randint(0, n_users - 1)  # rnd index
            for j in range(n_items):
                if R[i, j]:
                    temp_diff = R[i, j] - P[i].dot(Q[j])[0]

                    D_Pi = temp_diff * Q[:, j] - reg_lambda * P[i, :]
                    P[i, :] += learning_rate * D_Pi

                    # 直接向量运算就好啦
                    D_Qj = temp_diff * P[i, :] - reg_lambda * Q[:, j]
                    Q[:, j] += learning_rate * D_Qj
        
                    # 循环效率低了点，代码可读性不高
                    # for k in range(K):
                    #     D_P_ik = temp_diff * Q[j, k] - reg_lambda * P[i, k]
                    #     P[i, k] += learning_rate * D_P_ik
        
                    #     D_Q_jk = temp_diff * P[i, k] - reg_lambda * Q[j, k]
                    #     Q[j, k] += learning_rate * D_Q_jk

        # for epoch in range(max_epoch):
        #     i = random.randint(0, n_users - 1)  # rnd index
        #     Ai = R[i] != 0
        #     D_Pi = 2 * (Ai * (R[i] - np.dot(P[i], Q)).dot(Q.T)) - reg_lambda * np.sum(Ai) * P[i]
        #     D_Q = 2 * (Ai * (R[i] - P[i].dot(Q))).dot(P[i]) - reg_lambda * np.repeat(Ai, n_items).reshape(-1, n_items) * Q


        self.R_pred = P.dot(Q.T)
        return self.R_pred


    def predict(self, U):
        '''

        :param U: A one dimension ndarray containing a group of user_ids
        :return: The Rating ndarray
        '''
        R_pred = None
        for user_id in U.flatten():
            if R_pred is None:
                R_pred = self.R_pred[user_id]
            else:
                R_pred = np.vstack((R_pred, self.R_pred[user_id]))

        return R_pred

    def losses_estimate(self, R_train, R_val, K, learning_rate, max_epoch, reg_lambda,
                        min_loss_threshold=0.001):
        assert isinstance(R_train, np.ndarray)
        n_users, n_items = R_train.shape

        # 不能用0初始化...
        # P, Q = np.zeros((n_users, K)), np.zeros((n_items, K))
        P, Q = np.ones((n_users, K)), np.ones((n_items, K))

        train_losses = []
        val_losses = []

        for epoch in range(max_epoch):
            i = random.randint(0, n_users - 1)  # rnd index
            Ai_01 = R_train[i] != 0

            D_Pi = 2 * (Ai_01 * (R_train[i] - P[i] @ Q.T)).dot(Q) - reg_lambda * np.sum(Ai_01) * P[i]
            P[i] += learning_rate * D_Pi

            D_Q = 2 * (Ai_01 * (R_train[i] - P[i] @ Q.T)).T @ P[i] \
                  - reg_lambda * np.repeat(Ai_01, n_items).reshape(-1, n_items) * Q
            Q += learning_rate * D_Q

            curr_train_loss = MF_RMSE(R_train, P.T, Q.T)
            train_losses.append(curr_train_loss)

            curr_val_loss = MF_RMSE(R_val, P.T, Q.T)
            val_losses.append(curr_val_loss)

            if curr_train_loss < min_loss_threshold:
                break

        self.R_pred = P.dot(Q.T)
        return self.R_pred, train_losses, val_losses


class MF_ALS_Model:
    def __init__(self):
        pass


    def fit(self, R, K, reg_lambda, max_epoch, min_RMSE_threshold=0.1):
        """
        Fit a rating matrix and optimize the matrix factorization form using ALS method.

        Parameters
        ----------
        R : ndarray 
            The groundtruth rating matrix, in shape (n_users, n_items).
        K : int
            The number of latent features.
        reg_lambda : float
            The regularization parameter lambda of the model cost term.
        max_epoch : int
            The number of training epoches.
        min_cost_threshold : float
            When the training cost reaches or is lower than the thresold, training will stop.
        
        Returns
        -------
        R_pred : ndarray
            The predicted rating matrix.

        """
        assert isinstance(R, np.ndarray) and K > 0
        n_users, n_items = R.shape

        # 留意矩阵的维度
        U, M = np.random.random((K, n_users)), np.random.random((K, n_items))

        N_U = np.sum(R != 0, axis=1)  # 用户评分的次数 shape: (1)
        N_M = np.sum(R != 0, axis=0)  # 电影被评分的次数 shape: (1)
        # 用一部电影的平均评分作为该电影的第0个隐含特征的分数
        M[0, :] = np.sum(R, axis=0) / N_M  # shape: (2) <= shape: (2)
        # 有些电影在该数据集中没有被打分，相除后平均分数为无穷大，需要处理
        for i in range(n_items):
            # if M[0, i] == np.nan:
            if not (0 <= M[0, i] <= 5):
                M[0, i] = 0

        for epoch in range(max_epoch):
            # 必须把M_Ui, U_Mj这些小型矩阵抽解出来，而不是在庞大的原始矩阵上进行补0操作
            # 不然后面矩阵求逆会很麻烦，效率会很低，甚至因为0项太多，矩阵是不可逆的
            for i in range(n_users):
                M_Ui = None  # 把U[:, i]评价过的电影的特征列都挑选出来，组成一个(K * N_U[i])的小型矩阵
                R_Ui = None  # 把M_Ui对应的评分都挑选出来，组成一个(1 * N_U[i])的行向量
                for j in range(n_items):
                    if R[i, j]:
                        if M_Ui is not None:
                            M_Ui = np.hstack((M_Ui, M[:, j:j+1]))
                            R_Ui = np.hstack((R_Ui, R[i:i + 1, j:j + 1]))
                        else:
                            M_Ui = M[:, j:j+1]
                            R_Ui = R[i:i+1, j:j+1]

                # 有些用户在该数据集中没有评价任何电影
                if M_Ui is None:
                    continue

                Ai = M_Ui.dot(M_Ui.T) + reg_lambda * N_U[i] * np.eye(K)
                Vi = M_Ui.dot(R_Ui.T)

                U[:, i:i + 1] = np.dot(np.matrix(Ai).I.getA(), Vi)

            for j in range(n_items):
                U_Mj = None  # 把评价过电影M[j]的用户的喜好特征行挑选出来，组成一个(K * N_M[i])的小型矩阵
                R_Mj = None  # 把U_Mj对应的评分挑选出来 -- 一个(N_M[i] * 1)的列向量
                for i in range(n_users):
                    if R[i, j]:
                        if U_Mj is not None:
                            U_Mj = np.hstack((U_Mj, U[:, i:i+1]))
                            R_Mj = np.vstack((R_Mj, R[i:i+1, j:j+1]))
                        else:
                            U_Mj = U[:, i:i+1]
                            R_Mj = R[i:i+1, j:j+1]

                # 有些电影在该数据集中没有被任何用户评价
                if U_Mj is None:
                    continue

                Aj = np.dot(U_Mj, U_Mj.T) + reg_lambda * N_M[j] * np.eye(K)
                Vj = U_Mj.dot(R_Mj)

                M[:, j:j+1] = np.dot(np.matrix(Aj).I.getA(), Vj)

            curr_cost = MF_RMSE(R, U, M)
            if curr_cost <= min_RMSE_threshold:
                break

        self.U, self.M = U, M
        return np.dot(U.T, M)

    def cost_estimate(self, R_train, R_test, K, reg_lambda, max_epoch, min_RMSE_threshold=0.1):
        """
        Fit a rating matrix and optimize the matrix factorization form using ALS method.

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
        
        Returns
        -------
        R_pred : ndarray
            The predicted rating matrix.

        losses_dict : dict
            A dict containing the model's RMSE and MAE on training and testing dataset 
            during the training procedure.

        """
        assert isinstance(R_train, np.ndarray)
        n_users, n_items = R_train.shape

        # 留意矩阵的维度
        U, M = np.random.random((K, n_users)), np.random.random((K, n_items))

        N_U = np.sum(R_train != 0, axis=1)  # 用户评分的次数 shape: (1)

        N_M = np.sum(R_train != 0, axis=0)  # 电影被评分的次数 shape: (1)
        # 用一部电影的平均评分作为该电影的第0个隐含特征的分数
        M[0, :] = np.sum(R_train, axis=0) / N_M  # shape: (2) <= shape: (2)
        # 有些电影在该数据集中没有被打分，相除后平均分数为无穷大
        for i in range(n_items):
            # if M[0, i] == np.nan:
            if not (0 <= M[0, i] <= 5):
                M[0, i] = 0

        RMSE_losses_train, RMSE_losses_test = [], []
        MAE_losses_train, MAE_losses_test = [], []

        for epoch in range(max_epoch):

            # 必须把M_Ui, U_Mj这些小型矩阵抽解出来，而不是在庞大的原始矩阵上进行补0操作
            # 不然后面矩阵求逆会很麻烦，效率会很低，甚至因为0项太多，矩阵是不可逆的

            for i in range(n_users):
                M_Ui = None  # 把U[i]评价过的电影的特征列都挑选出来，组成一个(K * N_U[i])的小型矩阵
                R_Ui = None  # 把M_Ui对应的评分都挑选出来，组成一个(1 * N_U[i])的行向量
                for j in range(n_items):
                    if R_train[i, j]:
                        if M_Ui is not None:
                            M_Ui = np.hstack((M_Ui, M[:, j:j + 1]))
                            R_Ui = np.hstack((R_Ui, R_train[i:i + 1, j:j + 1]))
                        else:
                            M_Ui = M[:, j:j + 1]
                            R_Ui = R_train[i:i + 1, j:j + 1]

                # 有些用户在该数据集中没有评价任何电影
                if M_Ui is None:
                    continue

                Ai = M_Ui.dot(M_Ui.T) + reg_lambda * N_U[i] * np.eye(K)
                Vi = M_Ui.dot(R_Ui.T)

                U[:, i:i+1] = np.dot(np.matrix(Ai).I.getA(), Vi)

            for j in range(n_items):
                U_Mj = None  # 把评价过电影M[j]的用户的喜好特征行挑选出来，组成一个(K * N_M[i])的小型矩阵
                R_Mj = None  # 把U_Mj对应的评分挑选出来 -- 一个(N_M[i] * 1)的列向量
                for i in range(n_users):
                    if R_train[i, j]:
                        if U_Mj is not None:
                            U_Mj = np.hstack((U_Mj, U[:, i:i + 1]))
                            R_Mj = np.vstack((R_Mj, R_train[i:i + 1, j:j + 1]))
                        else:
                            U_Mj = U[:, i:i + 1]
                            R_Mj = R_train[i:i + 1, j:j + 1]

                # 有些电影在该数据集中没有被任何用户评价
                if U_Mj is None:
                    continue

                Aj = np.dot(U_Mj, U_Mj.T) + reg_lambda * N_M[j] * np.eye(K)
                Vj = U_Mj.dot(R_Mj)

                M[:, j:j + 1] = np.dot(np.matrix(Aj).I.getA(), Vj)

            curr_RMSE_train = MF_RMSE(R_train, U, M)
            curr_RMSE_test = MF_RMSE(R_test, U, M)

            RMSE_losses_train.append(curr_RMSE_train)
            RMSE_losses_test.append(curr_RMSE_test)

            curr_MAE_train = MF_MAE(R_train, U, M)
            curr_MAE_test = MF_MAE(R_test, U, M)

            MAE_losses_train.append(curr_MAE_train)
            MAE_losses_test.append(curr_MAE_test)

            if curr_RMSE_train <= min_RMSE_threshold:
                break

        self.U, self.M = U, M
        R_pred = np.dot(U.T, M)

        losses_dict = {
            'RMSE_Losses_train': RMSE_losses_train,
            'RMSE_Losses_test': RMSE_losses_test,
            'MAE_Losses_train': MAE_losses_train,
            'MAE_Losses_test': MAE_losses_test
        }

        return R_pred, losses_dict


def MF_RMSE(R, U, M):
    '''
    Evaluate the RMSE (Root Mean Square Error) of groundtruth rating matrix and the matrix factorization model.

    Parameters
    ----------
    R : ndarray
        The groundtruth rating matrix, an ndarray of shape(M, N), 
        where R[i, j] denotes the rating user i marks movie j.
    U : ndarray
        The User-Feature matrix, an ndarray of shape(K, M), 
        where U[k, i] denotes user i's preferred score for latent feature k.
    M : ndarray
        The Feature-Movie matrix, an ndarray of shape(K, N), 
        where M[k, j] denotes movie j's score for latent feature k.

    Returns
    -------
    RMSE : float
        The RMSE between R and np.dot(U.T, M)
    '''
    A_01 = R != 0
    temp_diff = A_01 * (R - np.dot(U.T, M))
    # return LA.norm(temp_diff, 2)
    return math.sqrt(np.sum((temp_diff * temp_diff).flatten()) / np.sum(A_01)) # 除以非0项的数量

def MF_MAE(R, U, M):
    '''
    Evaluate the MAE (Mean Absolute Error) of groundtruth rating matrix and the matrix factorization model.

    Parameters
    ----------
    R : ndarray
        The groundtruth rating matrix, an ndarray of shape(M, N), 
        where R[i, j] denotes the rating user i marks movie j.
    U : ndarray
        The User-Feature matrix, an ndarray of shape(K, M), 
        where U[k, i] denotes user i's preferred score for latent feature k.
    M : ndarray
        The Feature-Movie matrix, an ndarray of shape(K, N), 
        where M[k, j] denotes movie j's score for latent feature k.

    Returns
    -------
    MAE : float
        The MAE between R and np.dot(U.T, M)
    '''
    A_01 = R != 0
    temp_diff = A_01 * (R - np.dot(U.T, M))
    return np.sum(np.abs(temp_diff)) / np.sum(A_01)
