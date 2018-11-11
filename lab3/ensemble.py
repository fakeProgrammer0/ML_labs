import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.
    Only support binary classification currently.
    '''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_clf = weak_classifier
        self.n_weakers_limit = n_weakers_limit

        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples, n_features = X.shape
        w = np.ones(y.shape)

        weight_a = []
        self.clfs = []

        for i in range(self.n_weakers_limit):
            base_clf = self.weak_clf()
            base_clf.fit(X, y, w)
            self.clfs.append(base_clf)

            y_pred = base_clf.predict(X)

            # can be optimized into more clean and simple code
            # temp = np.ones(y.shape)
            # for j in range(n_samples):
            #     if y_pred[j][0] == y[j][0]:
            #         temp[j][0] = 0
            #
            # err_rate = w.dot(temp)[0] / w.sum()

            err_rate = w.dot(y_pred==y)[0] / w.sum()

            weight_param_a = math.log((1 - err_rate) / err_rate) / 2
            weight_a.append(weight_param_a)

            w = w * np.exp(-weight_param_a * y * y_pred)

            # prevent overfiting
            # if self.is_good_enough():
            #     break;

        self.a = np.ndarray((len(weight_a), 1), weight_a)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_score_pred = np.shape((X.shape[0], 1))
        # for i in range(len(self.clfs)):
        #     y_score_pred += self.a[i] * self.clfs[i].predict(X)
        for i, clf in enumerate(self.clfs):
            y_score_pred += self.a[i] * clf.predict(X)

        return y_score_pred

    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        def my_sign(a, threshold=0, sign_threshold=1):
            if sign_threshold != 1 or sign_threshold != -1:
                raise ValueError('sign_threshold must be -1 or 1')
            if not (-1 < threshold < 1):
                raise ValueError('threshold must be between -1 and 1')
            sign_a = np.zeros(a.shape)
            sign_a += a > threshold
            sign_a -= a < threshold
            sign_a += sign_threshold * (a == threshold)
            return sign_a

        return my_sign(self.predict_scores(X),threshold=threshold)

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
