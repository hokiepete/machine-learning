# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:46:38 2018

@author: pnola
"""

'''
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
'''
import pickle
#from sklearn.datasets import fetch_mldata
#pickle.dump(mnist, open( "mnist.pickle", "wb" ) )
with open('mnist.pickle', 'rb') as data:
    mnist = pickle.load(data)
    
x,y = mnist['data'], mnist['target']

import matplotlib
import matplotlib.pyplot as plt

some_digit = x[36000]
some_digit_image = some_digit.reshape(28,28)
'''
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()
'''
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
import numpy as np
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
#print sgd_clf.predict([some_digit])
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(x_train,y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = (y_train_5[train_index])
    x_test_fold = x_train[test_index]
    y_test_fold = (y_train_5[test_index])
    
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print n_correct/(len(y_pred)*1.0) 
'''
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X),1),dtype=bool)
    
never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5,y_train_pred))

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5,y_train_pred))
print(recall_score(y_train_5,y_train_pred))

from sklearn.metrics import f1_score
print(f1_score(y_train_5,y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold=0
y_some_digit_pred = (y_scores>threshold)
print(y_some_digit_pred)

threshold=200000
y_some_digit_pred = (y_scores>threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf,x_train,y_train_5,cv=3,method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],"b--",label="precision")
    plt.plot(thresholds, recalls[:-1],'g-',label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.show()

y_train_pred_90 = (y_scores>70000)
print(precision_score(y_train_5,y_train_pred_90))
print(recall_score(y_train_5,y_train_pred_90))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5,y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    
#plot_roc_curve(fpr,tpr)
#plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5,y_scores))

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,x_train,y_train_5,cv=3,method="predict_proba")
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
'''
plt.plot(fpr,tpr,'b:',label='SGD')
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend()
'''
print(roc_auc_score(y_train_5,y_scores_forest))

sgd_clf.fit(x_train,y_train)
print(sgd_clf.predict([some_digit]))

some_digit_scores =sgd_clf.decision_function([some_digit])
print(some_digit_scores)
print(np.argmax(some_digit_scores))
print(sgd_clf.classes_)
print(sgd_clf.classes_[5])

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(x_train,y_train)
print(ovo_clf.predict([some_digit]))
print(len(ovo_clf.estimators_))
forest_clf.fit(x_train,y_train)
print(forest_clf.predict([some_digit]))
print(forest_clf.predict_proba([some_digit]))

print(cross_val_score(sgd_clf,x_train,y_train,cv=3,scoring="accuracy"))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
print(cross_val_score(sgd_clf,x_train_scaled,y_train,cv=3,scoring="accuracy"))

y_train_pred = cross_val_predict(sgd_clf,x_train_scaled,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)
print(conf_mx)
#plt.matshow(conf_mx,cmap=plt.cm.gray)

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx,0)
#plt.matshow(norm_conf_mx,cmap=plt.cm.gray)

cl_a, cl_b = 3,5
x_aa = x_train[(y_train == cl_a) & (y_train_pred == cl_a)]
x_ab = x_train[(y_train == cl_a) & (y_train_pred == cl_b)]
x_ba = x_train[(y_train == cl_b) & (y_train_pred == cl_a)]
x_bb = x_train[(y_train == cl_b) & (y_train_pred == cl_b)]
'''
plt.figure(figsize=(8,8))
plt.subplot(221); plt.imshow(x_aa[:25],images_per_row=5)
plt.subplot(222); plot_digits(x_ab[:25],images_per_row=5)
plt.subplot(223); plot_digits(x_ba[:25],images_per_row=5)
plt.subplot(224); plot_digits(x_bb[:25],images_per_row=5)
'''
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large,y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train,y_multilabel)

print(knn_clf.predict([some_digit]))
#y_train_knn_pred = cross_val_predict(knn_clf,x_train,y_train,cv=3)
#print(f1_score(y_train,y_train_knn_pred,average='macro'))

noise = np.rnd.randint(0,100,(len(x_train),784))
noisetest = np.rnd.randint(0,100,(len(x_test),784))
x_train_mod = x_train + noise
x_test_mod = x_test + noisetest
y_train_mod = x_train
y_test_mod = x_test

knn_clf.fit(x_train_mod,y_train_mod)
clean_digit = knn_clf.predict([x_test_mod[some_digit]])
plt.imshow(clean_digit)













































