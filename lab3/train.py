from sklearn.tree import DecisionTreeClassifier

from lab3.ensemble import AdaBoostClassifier
from lab3.feature import NPDFeature
import cv2
import glob
import numpy as np
import os

def preprocess_imgs(dataset_dir, label):
    img_paths = glob.glob(dataset_dir + r'\*')
    X = None
    for img_path in img_paths:
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (24, 24))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_feat = NPDFeature(img).extract()

        if X is None:
            X = img_feat
        else:
            X = np.vstack((X, img_feat))

    y = np.array([label]*X.shape[0]).reshape(-1, 1)

    return X, y

def load_imgs():
    X_face, y_face = preprocess_imgs(os.getcwd() + r'\datasets\original\face', 1)



    pass


def test_my_adaboostClf(X, y):
    clf = AdaBoostClassifier(DecisionTreeClassifier, 10)
    clf.fit(X, y)



def test_sklearn_adaboostClf():
    pass

if __name__ == "__main__":
    # write your code here
    # test_my_adaboostClf()
    load_imgs()
    pass

