import numpy
import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784' , version=1)

x = mnist["data"].values

y = mnist["target"].values


# get the image, can return image as array or the plot of it. 
def get_image_square(index , plot = False):
    n = int(np.sqrt(len(x[index])))
    img = x[index].reshape(n,n)
    
    if plot is False:
        return img
    elif plot is True: 
        plt.title("plot of digit: {}".format(y[index]))
        plt.imshow(img , cmap  = 'binary')
    

get_image_square(1,plot = True)


number = 1

# Binary Classifier, verify if the data belongs or not to a specific class. 
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size = 0.3)
Y_train_n = ( Y_train == str(number) ) 

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train , Y_train_n)

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf , X_train , Y_train_n , cv = 3 , scoring = "accuracy")


def plot_number_is(k):  
    prediction = sgd_clf.predict([X_train[k]])
    plt.title("Number is {}? {} //// Predicted = {}".format(number , Y_train_n[k] , prediction))
    plt.imshow(X_train[k].reshape(28,28),cmap = 'binary')


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train_n , sgd_clf.predict(X_train))
# [43457,    44]
# [  341,  5158]

y_train_pred = sgd_clf.predict(X_train)

from sklearn.metrics import precision_score , recall_score 
from sklearn.metrics import classification_report

precision_score(Y_train_n , y_train_pred)
recall_score(Y_train_n , y_train_pred)

classification_report(Y_train_n , y_train_pred)
"""
              precision    recall  f1-score   support

       False       0.99      1.00      1.00     43501
        True       0.99      0.94      0.96      5499

    accuracy                           0.99     49000
   macro avg       0.99      0.97      0.98     49000
weighted avg       0.99      0.99      0.99     49000
"""
















