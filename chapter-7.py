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

