import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from Plot import plot_confusion_matrix
import matplotlib.pyplot as plt
from LoadSplit import LoadData
from LoadSplit import DataSplit
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score



# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names
# print(class_names[1])


df = pd.read_csv("heart.csv")
    
X=df[df.columns[0:13]]
y=df['target']
class_names= df.target_names
print(class_names)

# print(X.head())
# print(y.head())

# X, y=shuffle(X,y)

# print(X.head())
# print(y.head())

# cv = KFold(n_splits=20)
# clf = SVC(gamma='auto')
# y_pred = cross_val_predict(clf, X, y, cv=cv )
# print(accuracy_score(y, y_pred))
