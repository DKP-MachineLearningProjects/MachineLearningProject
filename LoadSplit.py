import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def LoadData():
    
    dt = datasets.load_digits()

    X = dt.data
    y = dt.target

    X,y =shuffle (X,y)
    class_names=dt.target_names

    # df = pd.read_csv("heart.csv")
    # X=df[df.columns[0:13]]
    # y=df['target']
      
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, class_names


def DataSplit(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)
    return X_train, X_test, y_train, y_test

LoadData()