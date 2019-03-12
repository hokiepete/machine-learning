# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:54:43 2018

@author: pnola
"""


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
x = iris['data'][:,(2,3)] #petal length / petak width
y = (iris['target']==2).astype(np.float64) #iris virginica

svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C=1,loss='hinge'))
        ))
svm_clf.fit(x,y)
print(svm_clf.predict([[5.5, 1.7]]))


from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
x,y = make_moons()
polynomial_svm_clf = Pipeline((
        ('poly_features', PolynomialFeatures(degree=3)),
        ('scalar', StandardScaler()),
        ('svm_clf',LinearSVC(C=10, loss='hinge'))
        ))

polynomial_svm_clf.fit(x,y)

from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
        ('scaler',StandardScaler()),
        ('svm_clf',SVC(kernel='poly',degree=3,coef0=1,C=5))
        ))
poly_kernel_svm_clf.fit(x,y)

rbf_kernel_svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('svm_clf',SVC(kernel='rbf',gamma=5,C=0.001))
        ))
rbf_kernel_svm_clf.fit(x,y)

from sklearn.svm import LinearSVR
svm_reg = Pipeline((
        ('scaler', StandardScaler()),
        ('svm',LinearSVR(epsilon=1.5))
        ))
svm_reg.fit(x,y)

from sklearn.svm import SVR
svm_poly_reg = Pipeline((
        ('scaler', StandardScaler()),
        ('svm_poly',SVR(kernel='poly',degree=2,C=100,epsilon=0.1))
        ))
svm_poly_reg.fit(x,y)

















