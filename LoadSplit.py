import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def LoadData():
    digits = datasets.load_digits()
    class_names = digits.target_names
    random = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    random.shuffle(indices)

    X = digits.data
    y = digits.target

    return X, y, class_names

def DataSplit(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)
    return X_train, X_test, y_train, y_test
