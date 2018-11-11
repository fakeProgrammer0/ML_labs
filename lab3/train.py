from sklearn.tree import DecisionTreeClassifier

from .ensemble import AdaBoostClassifier

if __name__ == "__main__":
    # write your code here

    clf = AdaBoostClassifier(DecisionTreeClassifier, 10)


    pass

