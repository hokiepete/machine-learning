# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:50:26 2018

@author: pnola
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
x = 2*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)
x_b=np.c_[np.ones((100,1)),x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
x_new = np.array([[0],[2]])
x_new_b = np.c_[np.ones((2,1)),x_new]
y_predict = x_new_b.dot(theta_best)
#plt.plot(x_new,y_predict,'r-')
#plt.plot(x,y,'b.')

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
#plt.plot(x_new,lin_reg.predict(x_new),'r-')
#plt.plot(x,y,'b.')

eta=0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
for interation in range(n_iterations):
    gradients = 2/m*x_b.T.dot(x_b.dot(theta)-y)
    theta = theta - eta*gradients
print(theta)

n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0 / (t+t1)
theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
print(theta)

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50,penalty=None,eta0=0.01)
sgd_reg.fit(x,y.ravel())
        
m = 100
x = 6*np.random.rand(m,1)-3
y = 0.5*x**2+x+2+np.random.randn(m,1)
#plt.plot(x,y,'.')
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2,include_bias=False)
x_poly = poly_features.fit_transform(x)

lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
#plt.plot(x,lin_reg.predict(x_poly),'r.')
#plt.plot(x,y,'b.')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curve(model,X,y):
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),'b-',linewidth=3,label="val")
    plt.legend()
    plt.ylim([0,3])
'''
lin_reg = LinearRegression()
plot_learning_curve(lin_reg,x,y)
'''
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
        ("poly_features",PolynomialFeatures(degree=3,include_bias=False)),
        ("sgd_reg",LinearRegression()),
        ))
#plot_learning_curve(polynomial_regression,x,y)

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1.0,solver='cholesky')
ridge_reg.fit(x,y)
print(ridge_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(x,y.ravel())
print(sgd_reg.predict([[1.5]]))

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x,y)
print(lasso_reg.predict([[1.5]]))

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1,l1_ratio=0.5)
elastic_net.fit(x,y)
print(elastic_net.predict([[1.5]]))

'''
from sklearn.base import clone
sgd_reg = SGDRegressor(n_iter=1,warm_start=True,penalty=None,learning_rate="constant",eta0=0.0005)
minimum_val_error = float("inf")
best_epoch=None
best_model=None
for epoch in range(1000):
    sgd_reg.fit(x,y)
    y_val_predict = sgd_reg.predict(x)
    val_error = mean_squared_error(y_val_predict,y)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch=epoch
        best_model=clone(sgd_reg)
'''
from sklearn import datasets
iris = datasets.load_iris()
x = iris["data"][:,3:]
y = (iris["target"] == 2).astype(np.int)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x,y)
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(x_new)
'''
plt.plot(x_new,y_proba[:,1],'g-',label="Iris-Virginica")
plt.plot(x_new,y_proba[:,0],'b--',label="Not Iris-Virginica")
plt.legend()
'''

x = iris["data"][:,(2,3)]
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10)
softmax_reg.fit(x,y)






#'''















