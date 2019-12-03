import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, KFold

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Plot import plot_confusion_matrix
import matplotlib.pyplot as plt
from LoadSplit import LoadData
from LoadSplit import DataSplit

def SVM_train_predict(X_train, X_test, y_train, k='linear'):
    svclassifier = SVC(kernel=k, gamma='auto')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return y_pred

def NB_train_predict(X_train, X_test, y_train):
    NB=GaussianNB()
    NB.fit(X_train, y_train)
    y_pred = NB.predict(X_test)
    return y_pred

def _main_():

    #loads the data from digits dataset and seperate the input features and target classes
    X,y, class_names = LoadData()
    
    #seperate the given dataset into training and testing samples
    X_train, X_test, y_train, y_test = DataSplit(X,y)
    
    #Accuracy_list holds the value of accuracy for each iteration and used to calculate the average accuracy
    Accuracy_list=[]
    #this controls the number of loop for each classifier.
    Total_Loop=20

    #This section is for Naive Bayes classifier without k-fold cross validation.
    for i in range(Total_Loop):
        X_train, X_test, y_train, y_test = DataSplit(X,y)
        y_pred = NB_train_predict(X_train, X_test, y_train)
        Accuracy_list.append(accuracy_score(y_test, y_pred))
    print("Accuracy of NB=", np.average(Accuracy_list))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix for NB')

    Accuracy_list.clear()

    #This section is for Naive Bayes classifier with k-fold cross validation.
    for i in range(Total_Loop):
        cv = KFold(n_splits=20)
        clf = GaussianNB()
        y_pred = cross_val_predict(clf, X, y, cv=cv )
        Accuracy_list.append(accuracy_score(y, y_pred))
    print("Accuracy of NB with Cross Validation=", np.average(Accuracy_list))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix NB with Cross Validation')

    Accuracy_list.clear()

    #This section is for Support Vector Classifier without k-fold cross validation.
    for i in range(Total_Loop):
        y_pred=SVM_train_predict(X_train, X_test, y_train, k='linear')
        Accuracy_list.append(accuracy_score(y_test, y_pred))
    print("Accuracy of SVM=", np.average(Accuracy_list))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix SVM')
    
    Accuracy_list.clear()
    #This section is for Naive Bayes classifier with k-fold cross validation.
    for i in range(Total_Loop):
        cv = KFold(n_splits=20)
        clf = SVC(kernel='linear',gamma='auto')
        y_pred = cross_val_predict(clf, X, y, cv=cv )
        Accuracy_list.append(accuracy_score(y,y_pred))
    print("Accuracy of SVM with Cross Validation=", np.average(Accuracy_list))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix SVM with Cross Validation')

    plt.show()

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names,title='Confusion matrix, without normalization')

    # print(classification_report(y_test,y_pred))
  

_main_()
