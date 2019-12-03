#import numpy as np
#import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

#this function is used for loading and seperating given dataset.
def LoadData():
    # load the digit dataset from sklearn.datasets.load_digits    
    dt = datasets.load_digits()

    #seperate the input features and target class into X and y
    X = dt.data
    y = dt.target

    #suffle the values in X and y randomly using sklearn.utils
    X,y =shuffle (X,y)
    class_names=dt.target_names

    return X, y, class_names


def DataSplit(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)
    return X_train, X_test, y_train, y_test

#for test purpose only
LoadData()