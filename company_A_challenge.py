import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, f1_score

encoder = LabelEncoder()
mean_imputer = SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)

lending = pd.read_csv('DR_Demo_Lending_Club.csv',header=0, sep=',')

#Drop unnecessary columns
lending.drop('Notes',axis=1,inplace=True)
lending.drop('purpose',axis=1,inplace=True)

#Remove NaNs and convert string data to numerical
emp_title = lending['emp_title']
emp_title.loc[emp_title.isnull()]='unknown'
lending['emp_title'] = encoder.fit_transform(emp_title)
del emp_title

emp_length = lending['emp_length']
emp_length.loc[emp_length=='na'] = '0'
lending['emp_length'] = encoder.fit_transform(emp_length)
del emp_length

earliest_cr_line = lending['earliest_cr_line']
earliest_cr_line.loc[earliest_cr_line.isnull()] = '0'
lending['earliest_cr_line'] = encoder.fit_transform(earliest_cr_line)
del earliest_cr_line

#Convert string data to numerical
lending['home_ownership'] = encoder.fit_transform(lending['home_ownership'])
lending['verification_status'] = encoder.fit_transform(lending['verification_status'])
lending['pymnt_plan'] = encoder.fit_transform(lending['pymnt_plan'])
lending['purpose_cat'] = encoder.fit_transform(lending['purpose_cat'])
lending['zip_code'] = encoder.fit_transform(lending['zip_code'])
lending['addr_state'] = encoder.fit_transform(lending['addr_state'])
lending['initial_list_status'] = encoder.fit_transform(lending['initial_list_status'])
lending['policy_code'] = encoder.fit_transform(lending['policy_code'])

#Replace NaNs from delinq and record data sets with (max + 1)
mths_since_last_delinq=lending['mths_since_last_delinq']
mths_since_last_delinq.loc[mths_since_last_delinq.isnull()]=mths_since_last_delinq.max()+1
del mths_since_last_delinq

mths_since_last_record = lending['mths_since_last_record']
mths_since_last_record.loc[mths_since_last_record.isnull()]=mths_since_last_record.max()+1
del mths_since_last_record

#Impute NaNs in rest of the data
mean_imputer.fit(lending)
x = mean_imputer.transform(lending)
lending = pd.DataFrame(x,columns=lending.columns)
del x

correlation_matrix = lending.corr()

y = lending['is_bad'].values
lending.drop('is_bad',axis=1,inplace=True)
x = lending.values

x_train, x_hold, y_train, y_hold = train_test_split(x,y,test_size=.2,random_state=42)

#Set up 5 CV folds
skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

#Logistic Regression
logreg = LogisticRegression(solver='liblinear')

#Custom cross validation
lr_cv_accuracy = []
lr_cv_f1 = []
lr_cv_log_loss = []
for train_index, test_index in skf.split(x_train,y_train):
    clone_logreg = clone(logreg)
    x_train_fold = x_train[train_index]
    x_test_fold = x_train[test_index]
    y_train_fold = y_train[train_index]
    y_test_fold = y_train[test_index]
    
    clone_logreg.fit(x_train_fold,y_train_fold)
    y_pred = clone_logreg.predict(x_test_fold)
    lr_cv_f1.append(f1_score(y_test_fold,y_pred))
    lr_cv_log_loss.append(log_loss(y_test_fold,y_pred))
    lr_cv_accuracy.append(sum(y_pred==y_test_fold)/len(y_pred))
    

#Built-in cross validation
from sklearn.model_selection import cross_validate
lr_scores = cross_validate(
        logreg, x_train, y_train, 
        scoring=('accuracy','f1','neg_log_loss'), cv=5)

logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_hold)
lr_final_f1 = f1_score(y_hold,y_pred)
lr_final_log_loss = log_loss(y_hold,y_pred)
lr_final_accuracy = sum(y_pred==y_hold)/len(y_pred)


#Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=1000)

#Custom cross validation
gb_cv_accuracy = []
gb_cv_f1 = []
gb_cv_log_loss = []
for train_index, test_index in skf.split(x_train,y_train):
    clone_gbc = clone(gbc)
    x_train_fold = x_train[train_index]
    x_test_fold = x_train[test_index]
    y_train_fold = y_train[train_index]
    y_test_fold = y_train[test_index]
    
    clone_gbc.fit(x_train_fold,y_train_fold)
    y_pred = clone_gbc.predict(x_test_fold)
    gb_cv_f1.append(f1_score(y_test_fold,y_pred))
    gb_cv_log_loss.append(log_loss(y_test_fold,y_pred))
    gb_cv_accuracy.append(sum(y_pred==y_test_fold)/len(y_pred))
    

#Built-in cross validation
from sklearn.model_selection import cross_validate
gb_scores = cross_validate(
        gbc, x_train, y_train, 
        scoring=('accuracy','f1','neg_log_loss'), cv=5)

gbc.fit(x_train,y_train)
y_pred = gbc.predict(x_hold)
gb_final_f1 = f1_score(y_hold,y_pred)
gb_final_log_loss = log_loss(y_hold,y_pred)
gb_final_accuracy = sum(y_pred==y_hold)/len(y_pred)


