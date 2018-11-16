from sklearn.tree import DecisionTreeClassifier

from lab3.ensemble import AdaBoostClassifier
from lab3.feature import NPDFeature
import cv2
import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

import pickle
dataset_dump_file = 'dataset.pickle'

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

    y = np.array([label]*X.shape[0]).reshape(-1, 1)
    D = np.hstack((y, X))

    return D
    # return X, y

def preprocess_imgs():
    D_face = load_imgs(os.getcwd() + r'\datasets\original\face', 1)
    D_nonface = load_imgs(os.getcwd() + r'\datasets\original\nonface', -1)

    D = np.vstack((D_face, D_nonface))
    with open(dataset_dump_file, 'wb') as f:
        # pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)

def load_divide_dataset():

    with open(dataset_dump_file, 'rb') as f:
        D = pickle.load(f)

    D_train, D_val = train_test_split(D, test_size=0.1)
    y_train, X_train = D_train[:, 0:1], D_train[:, 1:]
    y_val, X_val = D_val[:, 0:1], D_val[:, 1:]

    return X_train, y_train, X_val, y_val

def run_myadaboostClf():
    # preprocess_imgs()
    X_train, y_train, X_val, y_val = load_divide_dataset()

    X_train = X_train[0:1000, :]
    y_train = y_train[0:1000, :]


    clf = AdaBoostClassifier(DecisionTreeClassifier, 5)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    train_loss = zero_one_loss(y_true=y_train.flatten(),y_pred=y_train_pred.flatten())
    print('train_loss: ', train_loss)


    # DecisionTreeClassifier().fit()
    # DecisionTreeClassifier().predict()

def test_my_adaboostClf(X, y):
    clf = AdaBoostClassifier(DecisionTreeClassifier, 10)
    clf.fit(X, y)



def test_sklearn_adaboostClf():
    pass

if __name__ == "__main__":
    # write your code here
    run_myadaboostClf()

    pass

