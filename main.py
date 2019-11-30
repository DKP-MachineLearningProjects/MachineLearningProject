import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, KFold

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from Plot import plot_confusion_matrix
import matplotlib.pyplot as plt
from LoadSplit import LoadData
from LoadSplit import DataSplit

def SVM_train_predict(X_train, X_test, y_train, k='linear'):
    svclassifier = SVC(kernel=k, gamma='auto')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return y_pred

def KNN_train_predict(X_train, X_test, y_train):
    KNN=KNeighborsClassifier()
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    return y_pred

def NB_train_predict(X_train, X_test, y_train):
    NB=GaussianNB()
    NB.fit(X_train, y_train)
    y_pred = NB.predict(X_test)
    return y_pred

def DecisionTree_train_predict(X_train, X_test, y_train):
    DTC=DecisionTreeClassifier()
    DTC.fit(X_train, y_train)
    y_pred = DTC.predict(X_test)
    return y_pred


def main():

    X,y, class_names = LoadData()
    # X,y =LoadData()

    X_train, X_test, y_train, y_test = DataSplit(X,y)
    
    y_pred=SVM_train_predict(X_train, X_test, y_train, k='linear')
    print("Accuracy of SVM=", accuracy_score(y_test, y_pred))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix SVM')
    
    y_pred = NB_train_predict(X_train, X_test, y_train)
    print("Accuracy of NB=",accuracy_score(y_test, y_pred))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix for NB')

    cv = KFold(n_splits=15)
    clf = SVC(kernel='linear',gamma='auto')
    y_pred = cross_val_predict(clf, X, y, cv=cv )
    print("Accuracy of SVM with Cross Validation=", accuracy_score(y, y_pred))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix SVM with Cross Validation')

    plt.show()


    # y_pred=DecisionTree_train_predict(X_train, X_test, y_train)
    # np.set_printoptions(precision=2)
    # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix for DT')
    # plt.show()

  
    
    # # # Plot non-normalized confusion matrix
    # # plot_confusion_matrix(y_test, y_pred, classes=class_names,title='Confusion matrix, without normalization')

    # #   Plot normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

    
    # #print(classification_report(y_test,y_pred))
    

main()
