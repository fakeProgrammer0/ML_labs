import random
import numpy as np
from numpy import linalg as LA


class MF_SGD:
    def __init__(self):
        pass

    def fit(self, R, K, learning_rate, max_epoch, reg_lambda):
        assert R is np.ndarray
        n_users, n_items = R.shape

        P, Q = np.zeros((n_users, K)), np.zeros((n_items, K))

        # for epoch in range(max_epoch):
        #     i = random.randint(0, n_users - 1)  # rnd index
        #     for j in range(n_items):
        #         if R[i, j]:
        #             temp_diff = R[i, j] - P[i].dot(Q[j])[0]
        # 
        #             for k in range(K):
        #                 D_P_ik = 2 * temp_diff * Q[j, k] - reg_lambda * P[i, k]
        #                 P[i, k] += learning_rate * D_P_ik
        # 
        #                 D_Q_jk = 2 * temp_diff * P[i, k] - reg_lambda * Q[j, k]
        #                 Q[j, k] += learning_rate * D_Q_jk

        for epoch in range(max_epoch):
            i = random.randint(0, n_users - 1)  # rnd index
            Ai = R[i] != 0
            D_Pi = 2 * (Ai * (R[i] - np.dot(P[i], Q)).dot(Q.T)) - reg_lambda * np.sum(Ai) * P[i]
            D_Q = 2 * (Ai * (R[i] - P[i].dot(Q))).dot(P[i]) - reg_lambda * np.repeat(Ai, n_items).reshape(-1, n_items) * Q

        self.R_pred = P.dot(Q.T)
        return self.R_pred

    @staticmethod
    def cost(R, P, Q):
        A_01 = R != 0
        temp_diff = A_01 * (R - np.dot(P, Q.T))
        # return LA.norm(temp_diff, 2)
        return np.sum((temp_diff * temp_diff).flatten()) / R.size

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

        # for epoch in range(max_epoch):
        #     i = random.randint(0, n_users - 1)  # rnd index
        #     for j in range(n_items):
        #         if R_train[i, j]:
        #             err_diff = R_train[i, j] - P[i].dot(Q[j])
        #
        #             for k in range(K):
        #                 D_P_ik = 2 * err_diff * Q[j, k] - reg_lambda * P[i, k]
        #                 P[i, k] += learning_rate * D_P_ik
        #
        #                 D_Q_jk = 2 * err_diff * P[i, k] - reg_lambda * Q[j, k]
        #                 Q[j, k] += learning_rate * D_Q_jk

        for epoch in range(max_epoch):
            i = random.randint(0, n_users - 1)  # rnd index
            Ai = R_train[i] != 0

            D_Pi = 2 * (Ai * (R_train[i] - np.dot(P[i], Q.T))).dot(Q) - reg_lambda * np.sum(Ai) * P[i]
            P[i] += learning_rate * D_Pi

            D_Q = 2 * (Ai * (R_train[i] - P[i].dot(Q.T))).T.dot(P[i]) \
                  - reg_lambda * np.repeat(Ai, n_items).reshape(-1, n_items) * Q
            Q += learning_rate * D_Q

            curr_train_loss = MF_SGD.cost(R_train, P, Q)
            train_losses.append(curr_train_loss)

            curr_val_loss = MF_SGD.cost(R_val, P, Q)
            val_losses.append(curr_val_loss)

            if curr_train_loss < min_loss_threshold:
                break

        self.R_pred = P.dot(Q.T)
        return self.R_pred, train_losses, val_losses


class MF_ALS_Model:
    def __init__(self):
        pass


    def fit(self, R, K, reg_lambda, max_epoch, min_cost_threshold=0.001):
        assert isinstance(R, np.ndarray)
        n_users, n_items = R.shape

        U, M = np.random.random((n_users, K)), np.random.random((K, n_items))

        N_U = np.sum(R != 0, axis=1)  # 用户评分的次数 shape: (1)

        N_M = np.sum(R != 0, axis=0)  # 电影被评分的次数 shape: (1)
        # 用一部电影的平均评分作为该电影的第0个隐含特征的分数
        M[0, :] = np.sum(R, axis=0) / N_M  # shape: (2) <= shape: (2)
        # 有些电影在该数据集中没有被打分，相除后平均分数为无穷大
        for i in range(n_items):
            # if M[0, i] == np.nan:
            if not (0 <= M[0, i] <= 5):
                 M[0, i] = 0

        for epoch in range(max_epoch):

            for i in range(n_users):
                M_Ui = None  # 把U[i]评价过的电影的特征列都挑选出来，组成一个(K * N_U[i])的小型矩阵
                R_Ui = None  # 把M_Ui对应的评分都挑选出来，组成一个(1 * N_U[i])的行向量
                for j in range(n_items):
                    if R[i, j]:
                        if M_Ui is not None:
                            M_Ui = np.hstack((M_Ui, M[:, j:j+1]))
                            R_Ui = np.hstack((R_Ui, R[i:i + 1, j:j + 1]))
                        else:
                            M_Ui = M[:, j:j+1]
                            R_Ui = R[i:i+1, j:j+1]


                Ai = M_Ui.dot(M_Ui.T) + reg_lambda * N_U[i] * np.eye(K)
                Vi = M_Ui.dot(R_Ui.T)

                U[i] = np.dot(np.matrix(Ai).I.getA(), Vi).flatten()

            for j in range(n_items):
                U_Mj = None  # 把评价过电影M[j]的用户的喜好特征行挑选出来，组成一个(N_M[i] * K)的小型矩阵
                R_Mj = None  # 把U_Mj对应的评分挑选出来 -- 一个(N_M[i] * 1)的列向量
                for i in range(n_users):
                    if R[i, j]:
                        if U_Mj is not None:
                            U_Mj = np.vstack((U_Mj, U[i:i+1, :]))
                            R_Mj = np.vstack((R_Mj, R[i:i+1, j:j+1]))
                        else:
                            U_Mj = U[i:i+1, :]
                            R_Mj = R[i:i+1, j:j+1]

                Aj = np.dot(U_Mj.T, U_Mj) + reg_lambda * N_M[j] * np.eye(K)
                Vj = U_Mj.T.dot(R_Mj)

                M[:, j:j+1] = np.dot(np.matrix(Aj).I.getA(), Vj)

            curr_cost = MF_cost(R, U, M)
            if curr_cost <= min_cost_threshold:
                break

        self.U, self.M = U, M
        return U.dot(M)

    def cost_estimate(self, R_train, R_val, K, reg_lambda, max_epoch, min_cost_threshold=0.0001):
        assert isinstance(R_train, np.ndarray)
        n_users, n_items = R_train.shape

        U, M = np.random.random((n_users, K)), np.random.random((K, n_items))

        N_U = np.sum(R_train != 0, axis=1)  # 用户评分的次数 shape: (1)

        N_M = np.sum(R_train != 0, axis=0)  # 电影被评分的次数 shape: (1)
        # 用一部电影的平均评分作为该电影的第0个隐含特征的分数
        M[0, :] = np.sum(R_train, axis=0) / N_M  # shape: (2) <= shape: (2)
        # 有些电影在该数据集中没有被打分，相除后平均分数为无穷大
        for i in range(n_items):
            # if M[0, i] == np.nan:
            if not (0 <= M[0, i] <= 5):
                M[0, i] = 0

        losses_train, losses_val = [], []

        for epoch in range(max_epoch):

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

                U[i] = np.dot(np.matrix(Ai).I.getA(), Vi).flatten()

            for j in range(n_items):
                U_Mj = None  # 把评价过电影M[j]的用户的喜好特征行挑选出来，组成一个(N_M[i] * K)的小型矩阵
                R_Mj = None  # 把U_Mj对应的评分挑选出来 -- 一个(N_M[i] * 1)的列向量
                for i in range(n_users):
                    if R_train[i, j]:
                        if U_Mj is not None:
                            U_Mj = np.vstack((U_Mj, U[i:i + 1, :]))
                            R_Mj = np.vstack((R_Mj, R_train[i:i + 1, j:j + 1]))
                        else:
                            U_Mj = U[i:i + 1, :]
                            R_Mj = R_train[i:i + 1, j:j + 1]

                # 有些电影在该数据集中没有被任何用户评价
                if U_Mj is None:
                    continue

                Aj = np.dot(U_Mj.T, U_Mj) + reg_lambda * N_M[j] * np.eye(K)
                Vj = U_Mj.T.dot(R_Mj)

                M[:, j:j + 1] = np.dot(np.matrix(Aj).I.getA(), Vj)

            curr_cost_train = MF_cost(R_train, U, M)
            curr_cost_val = MF_cost(R_val, U, M)

            losses_train.append(curr_cost_train)
            losses_val.append(curr_cost_val)

            if curr_cost_train <= min_cost_threshold:
                break

        self.U, self.M = U, M
        return U.dot(M), losses_train, losses_val


def MF_cost(R, U, M):
    '''Evaluate the cost (total loss) of the matrix factorization model

    :param R: The Rating Matrix, an ndarray of shape(M, N), where R[i, j] denotes the rating user i marks movie j
    :param U: The User-Feature Matrix, an ndarray of shape(M, K), where U[i, k] denotes user i's preferred score for latent feature k
    :param M: The Feature-Movie Matrix, an ndarray of shape(K, N), where M[k, j] denotes movie j's score for latent feature k
    :return: the cost between R and UM
    '''
    A_01 = R != 0
    temp_diff = A_01 * (R - np.dot(U, M))
    # return LA.norm(temp_diff, 2)
    return np.sum((temp_diff * temp_diff).flatten()) / np.sum(A_01) # 除以非0项的数量