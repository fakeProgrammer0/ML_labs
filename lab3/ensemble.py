import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.
    Only support binary classification in which the label y is from {-1, +1} currently.
    '''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_clf = weak_classifier
        self.n_weakers_limit = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1),
               where the class label y[i, 0] is from {-1, +1}.
        '''
        w = np.ones(y.shape)
        w = w / w.sum() # regularization

        self.a = []
        self.base_clfs = []

        for i in range(self.n_weakers_limit):
            base_clf = self.weak_clf(max_depth=2)
            base_clf.fit(X, y.flatten(), w.flatten())

            y_pred = base_clf.predict(X).reshape((-1, 1))

            err_rate = w.T.dot(y_pred != y)[0][0] / w.sum()

            if err_rate > 1 / 2 or err_rate == 0.0:
                break

            weight_param_a = math.log((1 - err_rate) / err_rate) / 2

            self.base_clfs.append(base_clf)
            self.a.append(weight_param_a)

            w = w * np.exp(-weight_param_a * y * y_pred)
            w = w / w.sum() # regularization

            # prevent overfiting
            # if self.is_good_enough():
            #     break;


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_score_pred = np.zeros((X.shape[0], 1))
        for i, clf in enumerate(self.base_clfs):
            y_score_pred += self.a[i] * clf.predict(X).reshape((-1, 1))

        return y_score_pred

    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''

        def sign_helper(a, threshold=0, sign_threshold=1):
            ''' A helper method to perform the sign operation on each element of a given array a
            :param a: An ndarray
            :param threshold: The threshold of the sign function, default 0.
                                sign(x) = -1 where x < threshold
                                sign(x) = +1 where x > threshold
                                sign(x) = sign_threshold where x = threshold
            :param sign_threshold: default 1
            :return: The result of the sign operation on the given array a
            '''
            if sign_threshold != 1 and sign_threshold != -1:
                raise ValueError('sign_threshold must be -1 or +1')
            # if not (-1 < threshold < 1):
            #     raise ValueError('threshold must be between -1 and 1')
            sign_a = np.zeros(a.shape)
            sign_a += a > threshold
            sign_a -= a < threshold
            sign_a += sign_threshold * (a == threshold)
            return sign_a

        return sign_helper(self.predict_scores(X), threshold=threshold)

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
