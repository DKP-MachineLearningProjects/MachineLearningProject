#from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, confusion_matrix
from Plot import plot_confusion_matrix

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

def SVM_train_predict(X_train, X_test, y_train):
    svclassifier1 = SVC(kernel='linear')
    svclassifier1.fit(X_train, y_train)
    y_pred = svclassifier1.predict(X_test)
    return y_pred


def main():

    X,y, class_names = LoadData()
    X_train, X_test, y_train, y_test = DataSplit(X,y)
    y_pred=SVM_train_predict(X_train, X_test, y_train)
    
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names,title='Confusion matrix, without normalization')

    #   Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

    plt.show()

    #print(classification_report(y_test,y_pred))
    

main()
