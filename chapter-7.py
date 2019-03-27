from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import numpy as np
x,y = make_moons(n_samples=1000)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
        estimators=[('lr',log_clf), ('rf',rnd_clf), ('svc',svm_clf)],
        voting='soft'
        )
voting_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_val)
    print(clf.__class__.__name__,accuracy_score(y_val,y_pred))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),n_estimators=5,
        max_samples=100, bootstrap=True, n_jobs=1
        )
bag_clf.fit(x_train,y_train)
y_pred = bag_clf.predict(x_val)
print(bag_clf.__class__.__name__,accuracy_score(y_val,y_pred))

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=1)
rnd_clf.fit(x_train,y_train)
y_pred = rnd_clf.predict(x_val)
#print(y_pred)

from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=1)
rnd_clf.fit(iris["data"],iris["target"])
for name, score in zip(iris["feature_names"],rnd_clf.feature_importances_):
    print(name, score)

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5
        )
ada_clf.fit(x_train,y_train)
y_pred = ada_clf.predict(x_val)
#print(y_pred)

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(x,y)

y2 = y - tree_reg1.predict(x)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(x,y2)

y3 = y2 - tree_reg2.predict(x)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(x,y3)

y_pred = sum(tree.predict(x) for tree in (tree_reg1,tree_reg2,tree_reg3))

from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=3, learning_rate=1.0)
gbrt.fit(x,y)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

xtrain, xval, ytrain, yval = train_test_split(x,y)

gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=120)
gbrt.fit(xtrain,ytrain)

errors =[mean_squared_error(yval,ypred) for ypred in gbrt.staged_predict(xval)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(xtrain,ytrain)

gbrt = GradientBoostingRegressor(max_depth=2,warm_start=True)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1,320):
    gbrt.n_estimators = n_estimators
    gbrt.fit(xtrain,ytrain)
    ypred = gbrt.predict(xval)
    val_error = mean_squared_error(yval,ypred)
    if val_error < min_val_error:
        min_val_error=val_error
        error_going_up=0
    else:
        error_going_up+=1
        if error_going_up==5:
            break


