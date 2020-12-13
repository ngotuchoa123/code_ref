#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 08:25:55 2020

@author: quang
"""
import numpy as np
import pandas as pd
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
#%% Basic Information
print(data.shape[0])
print(data.shape[1])
print(data.columns)
print(data.info())
print(data.isnull().sum())
#%% 
print(data.head())
#%% count observation by store_nbr
print(data['Attrition'].value_counts())
print(data['JobSatisfaction'].value_counts())

#%% Show all columns
pd.set_option('display.max_columns', 100)
print(data.head(5))


#%% DATA VISUALIZATION
import matplotlib.pyplot as plt

#%% SEABORN LIBRARY
import seaborn as sns

sns.distplot(data['Education'], bins = 25)
#%%
sns.distplot(data['DistanceFromHome'], bins = 25)
#%% 2 Numerical
sns.jointplot(x = 'JobSatisfaction', y = 'Age', data = data)
#%%
sns.jointplot(x = 'JobSatisfaction', y = 'DistanceFromHome', data = data, kind='reg')
#%%
sns.jointplot(x = 'JobSatisfaction', y = 'Education', data = data, kind='kde')

data.JobSatisfaction.corr(data.Education)

sns.jointplot(x = 'JobSatisfaction', y = 'Education', data = data, kind='reg')

#%% Small dataset
sns.pairplot(data)
#%% 1, 2 Numerical + 1 Categorical
sns.pairplot(data, hue = 'Attrition')

#%% 1 Categorical
sns.countplot(x = 'Attrition', data = data)
sns.countplot(x = 'JobSatisfaction', data = data)
#%% 1 Categorical + 1 Numerical
sns.barplot(x = 'Attrition', y = 'Age', data = data)
#%% 1 Categorical + 1 Numerical
sns.boxplot(x = 'JobSatisfaction', y = 'Attrition', data = data)  # Outlier

#%% 2 Categorical + 1 Numerical
sns.boxplot(x = 'JobRole', y = 'JobSatisfaction', data = data, hue = 'Attrition')  # Outlier



#%% HOUSE DATASET
import pandas as pd
import numpy as np
data = pd.read_csv('house_price.csv')
#%% Basic Information
print(data.shape[0])  # 1460
print(data.shape[1])  # 81
print(data.columns)
print(data.info())
print(data.isnull().sum())
#%% Which variable has null?
print(data.isnull().sum() > 0)
print(data.columns[data.isnull().sum() > 0])
null_vars = data.columns[data.isnull().sum() > 0]
#%%
print(data[null_vars].isnull().sum())

print(data[null_vars].isnull().sum()/len(data))

data = data.drop(['Id','Alley','PoolQC','Fence','MiscFeature'], axis = True)
null_vars = data.columns[data.isnull().sum() > 0]
#%% Null transform for categorical data
null_cat_vars = [x for x in null_vars if data[x].dtypes == 'O']
print(null_cat_vars)
for x in null_cat_vars:
    data[x].fillna('Unknown', inplace = True)
#%% Null transform for numerical data
null_cont_vars = [x for x in null_vars if data[x].dtypes != 'O']
print(null_cont_vars)
for x in null_cont_vars:
    temp = np.nanmedian(data[x])
    data[x].replace(np.nan, temp, inplace = True)
#%% CHECK NULL - NO MORE NULL
print(data[null_vars].isnull().sum())


#%%
cat_vars = [x for x in data if data[x].dtypes == 'O']
import seaborn as sns
import matplotlib.pyplot as plt
for c in cat_vars:
    sns.countplot(x=c,data=data)
    plt.show()


for c in cat_vars:
    #print(c)
    #print(data[c].value_counts())
    temp = data[c].value_counts().sort_values(ascending=False)
    if len(temp[temp<=10])>0:
        for i in temp.index[temp<=10]:
            data[c].replace(i,temp.index[0],inplace=True)
        
for c in cat_vars:
    if len(data[c].value_counts()) == 1:
        data.drop(c,axis=1,inplace=True)

#%% EDA continous

num_vars = [x for x in data if data[x].dtypes != 'O']       
        
for c in num_vars:
           sns.distplot(data[c],bins=30,kde=False)
           plt.show()     
        
for c in num_vars:
    median_value = np.nanmedian(data[c])
    data[c].replace(np.nan, median_value, inplace = True)
           

data = pd.get_dummies(data,drop_first=True)



X = data.drop(['SalePrice'], axis = True)
y = data['SalePrice']


#########

#

cat_var = data.columns[data.dtypes == 'O']

print("UNDERSTANDING THE CATEGORICAL DISTRIBUTION BY THE TARGET (ATTRITION)")
print("NOTE: - It's a plot just about the columns with maximum 10 values.")
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18,35))
fig.subplots_adjust(hspace=0.5, bottom=0)
# fig.suptitle('BINARY FEATURES by the TARGET feature', fontsize=22)

for ax, catplot in zip(axes.flatten(), cat_var):
        sns.countplot(x=catplot, data=data, hue='Attrition', ax=ax, )
        ## GEting the ratio of Years with current manager just to test into graphs
        ax.set_title(catplot.upper(), fontsize=18)
        ax.set_ylabel('Count', fontsize=16)
        ax.set_xlabel(f'{catplot} Values', fontsize=15)
        ax.legend(title='Attrition', fontsize=12)
        
        
        
########3
num_var = data.columns[data.dtypes != 'O']
plt.figure(figsize=(20,15))
plt.title('Correlation of Num Var', fontsize=25)
sns.heatmap(data[num_var].astype(float).corr(), vmax=1.0 )
plt.show()       

for c in cat_var:
    sns.countplot(x=c,data=data)
    plt.show()
    
for c in num_var:
           sns.distplot(data[c],bins=30)
           plt.show() 


###### test F
           
data.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
num_var = data.columns[data.dtypes != 'O']
from sklearn.feature_selection import SelectKBest, f_classif

X_indices = np.arange(len(num_var))
selector = SelectKBest(f_classif, k=8)
selector.fit(data[num_var], data.Attrition)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
import matplotlib.pyplot as plt
plt.bar(X_indices - .45, scores, width=.2,label=r'Univariate score ($-Log(p_{value})$)')

num_var[3:]

f_classif(data[num_var], data.Attrition)


abc='Non-Travel'

dict(abc = 0, Travel_Rarely = 1,Travel_Frequently=2)


data.BusinessTravel =data.BusinessTravel.replace('Non-Travel','Non_Travel')

data.BusinessTravel.value_counts()

X = pd.get_dummies(data)
X.columns
X.info()

