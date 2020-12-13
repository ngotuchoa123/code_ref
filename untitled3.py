# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:13:43 2020

@author: Lenovo ThinkPad
"""

import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import pandas as pd
import numpy as np

dataset = pd.read_csv('base_all.csv')

dataset = dataset.loc[(dataset.month == 4) | (dataset.month == 3),]

dataset_oot = dataset.loc[(dataset.month == 5),]

dataset_oot = dataset_oot.drop(['Unnamed: 0','csn','month'], axis=1)

dataset_oot['target'] = np.multiply(dataset_oot['target'], 1) 

dataset_oot['target'].value_counts()

X_oot = dataset_oot.iloc[:,1:]
y_oot = dataset_oot.iloc[:,0]



dataset.columns

dataset = dataset.drop(['Unnamed: 0','csn','month'], axis=1)

dataset['target'] = np.multiply(dataset['target'], 1) 

dataset['target'].value_counts()


X = dataset.iloc[:,1:]
Y = dataset.iloc[:,0]


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


print(model)

y_pred = model.predict(X_test)
predictions =y_pred

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


auc(y_test, predictions)


predictions[1:4]


import numpy as np
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
metrics.auc(fpr, tpr)

plot(fpr, tpr)

my_model = XGBClassifier(n_estimators=10000)
my_model.fit(X_train, y_train, early_stopping_rounds=50, 
             eval_set=[(X_test, y_test)], verbose=False)

predictions = my_model.predict(X_test)
predictions2 = my_model.predict_proba(X_test)
predictions2 = [predictions2[i][1] for i in range(predictions2.shape[0])]


fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions2)
2*metrics.auc(fpr, tpr)-1



predictions2 = my_model.predict_proba(X_oot)
predictions2 = [predictions2[i][1] for i in range(predictions2.shape[0])]


fpr, tpr, thresholds = metrics.roc_curve(y_oot, predictions2)
metrics.auc(fpr, tpr)



fpr, tpr, thresholds = metrics.roc_curve(y_oot, predict_ex)
metrics.auc(fpr, tpr)

feature_important = my_model.get_booster().get_score(importance_type='weight')

keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.plot(kind='barh')

from xgboost import  plot_importance
plot_importance(my_model, max_num_features = 15)
pyplot.show()

my_model.get_booster().get_score()



#%%%%


import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 7

d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

watchlist = [(d_train, 'train'), (d_test, 'test')]

bst = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=50, verbose_eval=10)


plot_importance(bst, max_num_features = 15)
pyplot.show()





predictions2 = bst.predict_proba(X_oot)
predictions2 = [predictions2[i][1] for i in range(predictions2.shape[0])]


fpr, tpr, thresholds = metrics.roc_curve(y_oot, predictions2)
metrics.auc(fpr, tpr)

check = pd.DataFrame({'actual':y_oot,'predict':predictions2})

def quantile_dev(x,cutoff):
    if x>cutoff[0]:
        return 1
    elif x>cutoff[1]:
        return 2
    elif x>cutoff[2]:
        return 3
    elif x>cutoff[3]:
        return 4
    elif x>cutoff[4]:
        return 5
    elif x>cutoff[5]:
        return 6
    elif x>cutoff[6]:
        return 7
    elif x>cutoff[7]:
        return 8
    elif x>cutoff[8]:
        return 9
    else: return 10


cutoff = [0.851387,0.800354,0.722402,0.658771,0.571567,0.477107,0.432403,0.384042,0.326855]

quantile_dev(0.9,cutoff)

check.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9], axis = 0) 


new = s.apply(lambda num : num + 5) 

x = lambda a : a + 10

check['rank'] = check.predict.apply(lambda a : quantile_dev(a,cutoff))

res = check.groupby(['rank'])['actual', 'predict'].agg([('mean',np.mean),('sum',np.sum), ('count','count')]).reset_index()


d_oot = xgb.DMatrix(X_oot)
p_oot = bst.predict(d_oot)

fpr, tpr, thresholds = metrics.roc_curve(y_oot, p_oot)
2*metrics.auc(fpr, tpr)-1

print(my_model.feature_importances_)

import matplotlib.pyplot as pyplot

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

plot_importance(model)
pyplot.show()



###########
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        
        
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10, 15],
        'gamma': [0.5, 1, 1.5, 2, 5, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10]
        }


xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)








