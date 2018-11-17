'''Solving a face classification problem using AdaBoost method from ensemble.py
Attention: To exhibit the performance gained by using AdaBoost to aggregate weak learners, decision tree classifiers with max_depth = 1 are chosen as weak classifiers.
[为了展示“AdaBoost能够集成弱分类器，获得更好的性能”，使效果更加明显，这次实验故意采用了max_depth设置为1的决策树作为弱分类器]
'''

from lab3.ensemble import AdaBoostClassifier
from lab3.feature import NPDFeature

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report

import numpy as np
import cv2

import os
import glob
import pickle
from datetime import datetime

from lab3.ML_toolkit import exp_loss

dataset_dump_file = os.getcwd() + r'\dataset.pickle'
report_file = os.getcwd() + r'\report.txt'

def preprocess_imgs():
    '''load sample images, extract their NPD features and save the data into the local cache file 'dataset.pickle'
    '''

    def load_imgs(dataset_dir, label):
        img_paths = glob.glob(dataset_dir + r'\*')
        X = None
        for img_path in img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (24, 24))
            img_feat = NPDFeature(img).extract()

            if X is None:
                X = img_feat
            else:
                X = np.vstack((X, img_feat))

        y = np.array([label] * X.shape[0]).reshape(-1, 1)
        D = np.hstack((y, X))
        return D

    D_face = load_imgs(os.getcwd() + r'\datasets\original\face', 1)
    D_nonface = load_imgs(os.getcwd() + r'\datasets\original\nonface', -1)

    D = np.vstack((D_face, D_nonface))
    with open(dataset_dump_file, 'wb') as f:
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)

def load_divide_dataset(test_size):
    with open(dataset_dump_file, 'rb') as f:
        D = pickle.load(f)

    D_train, D_val = train_test_split(D, test_size=test_size)
    y_train, X_train = D_train[:, 0:1], D_train[:, 1:]
    y_val, X_val = D_val[:, 0:1], D_val[:, 1:]

    return X_train, y_train, X_val, y_val

def face_classification_adaboost():

    def classified_result(y_train_true, y_train_pred, y_val_true, y_val_pred, report_title='', report_file=report_file):
        '''A helper method to write classified result into report_file
        label shape: 1d ndarray
        '''

        train_loss_exp = exp_loss(y_train_true, y_train_pred)
        train_loss_01 = zero_one_loss(y_train_true, y_train_pred)

        val_loss_exp = exp_loss(y_val_true, y_val_pred)
        val_loss_01 = zero_one_loss(y_val_true, y_val_pred)

        with open(report_file, 'a+') as report_fp:
            report_fp.write(report_title+'\n')
            report_fp.write('timestamp: ' + str(datetime.now()) + '\n\n')

            report_fp.write('train_loss_exp = {:.6f}\n'.format(train_loss_exp))
            report_fp.write('train_loss_01  = {:.6f}\n'.format(train_loss_01))
            report_fp.write('val_loss_exp   = {:.6f}\n'.format(val_loss_exp))
            report_fp.write('val_loss_01    = {:.6f}\n\n'.format(val_loss_01))

            report_fp.write('classification_report of train data:\n')
            report_fp.write(classification_report(y_true=y_train_true, y_pred=y_train_pred,
                                                  target_names=class_name) + '\n\n')
            report_fp.write('classification_report of val data:\n')
            report_fp.write(classification_report(y_true=y_val_true, y_pred=y_val_pred,
                                                  target_names=class_name) + '\n\n')

    class_name = ['face', 'non-face']

    if dataset_dump_file not in glob.glob(os.getcwd() + r'\*'):
        preprocess_imgs()
    X_train, y_train, X_val, y_val = load_divide_dataset(test_size=0.25)

    # ------- A single weak classifier ------------

    weak_clf = DecisionTreeClassifier(max_depth=1)
    weak_clf.fit(X_train, y_train.flatten())

    y_train_pred = weak_clf.predict(X_train)
    y_val_pred = weak_clf.predict(X_val)

    classified_result(y_train.flatten(), y_train_pred, y_val.flatten(), y_val_pred, '1.loss estimate of a single weak classifier (a sklearn.tree.DecisionTreeClassifier with max_depth = 1):')

    # ------- AdaBoost ------------

    n_weak_classifier = 10

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_weak_classifier)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    classified_result(y_train.flatten(), y_train_pred.flatten(), y_val.flatten(), y_val_pred.flatten(), '2.loss estimate of AdaBoost (base classifier: sklearn.tree.DecisionTreeClassifier with max_depth = 1):')

def adaboost_loss_estimate():
    if dataset_dump_file not in glob.glob(os.getcwd() + r'\*'):
        preprocess_imgs()
    X_train, y_train, X_val, y_val = load_divide_dataset(test_size=0.25)

    n_weak_classifier = 10
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_weak_classifier)
    clf.loss_estimate(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    # write your code here
    face_classification_adaboost()
    # adaboost_loss_estimate()
    pass


