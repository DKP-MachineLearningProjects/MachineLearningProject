import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from Plot import plot_confusion_matrix
import matplotlib.pyplot as plt
from LoadSplit import LoadData
from LoadSplit import DataSplit

def SVM_train_predict(X_train, X_test, y_train):
    svclassifier1 = SVC(kernel='poly')
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
