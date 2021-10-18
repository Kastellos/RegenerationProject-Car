#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:31:20 2021

@author: mav24
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PowerTransformer
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.ensemble import ExtraTreesRegressor, IsolationForest

path_train = '/home/mav24/Documents/Development/Regeneration/Project/Data/training_data.xlsx'
data_train = pd.read_excel(path_train)
drop = ['Unnamed: 0', 'encoded car brand', 'station wagon', 'cylinders', 'encoded origin']
data_train.drop(columns=drop, inplace=True)

path_val = '/home/mav24/Documents/Development/Regeneration/Project/Data/validation_data.xlsx'
data_val = pd.read_excel(path_val)

data_val.drop(columns=drop, inplace=True)

data = pd.concat((data_train, data_val))


# Scaling the data Standar sceler
X = data.drop(columns='mpg')
Y = data['mpg']
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=13)


# Outliers Detection
iso = IsolationForest(contamination=0.05)
yhat = iso.fit_predict(X_train)

mask = yhat != -1
X_train, Y_train = X_train[mask, :], Y_train[mask]



# Sanity check of shapes
print ('Shape of X_train=>',X_train.shape)
print ('Shape of X_test=>',X_test.shape)
print ('Shape of Y_train=>',Y_train.shape)
print ('Shape of Y_test=>',Y_test.shape)


# Making the data and labels to lightgbm dataset
train_data=lgb.Dataset(X_train, label=Y_train)

# Setting validation dataset
validation_data=train_data.create_valid(X_test, label=Y_test)


# Setting hyperparameters for the lgb model
lgbm_parameters={'learning_rate':0.0001,
                'boosting_type':'gbdt',
                'objective':'regression',
                'metric':'rmse',
                'force_col_wise':True,
                'num_leaves':100,
                'max_depth':10,
                'verbose':-1}

clf=lgb.train(lgbm_parameters, train_data, 100000, valid_sets=[validation_data], early_stopping_rounds=100)
#x=clf.save_model()

y_pred_clf=clf.predict(X_test)

# Showing MSE and MAE
print ('Model: LightGBM classifier')
print ('Model squere of MSE=>{}'.format(mse(Y_test, y_pred_clf, squared=False)))
print ('Model MAE=>{}'.format(mae(Y_test, y_pred_clf)))
print (f'Model R2 score: {r2(Y_test, y_pred_clf)}\n\n')


xtr = ExtraTreesRegressor()

xtr.fit(X_train, Y_train)
xtr_pred_test = xtr.predict(X_test)


print('Model: Extra Trees Regressor')
print(f'The R2 accuracy is: {r2(Y_test, xtr_pred_test)}')
print(f'The mean square error is: {mse(Y_test, xtr_pred_test, squared=False)}')
print(f'Mean absolute error is: {mae(Y_test, xtr_pred_test)}')

pipe = Pipeline(steps=[('scaler', StandardScaler()),
                       ('extr', ExtraTreesRegressor(n_jobs=3))])

param_grid = {'extr__n_estimators':[100],
              #'extr__criterion':['squared_error', 'mse', 'mae'],
              'extr__max_depth':[None, 10, 20, 50, 100, 200, len(X_train)],
              #'extr__min_samples_split':[1,2,3,5,10],
              #'extr__min_samples_leaf':[1,2,3,5,10],
              'extr__max_features':['auto', 'sqrt', 'log2'],
              #'extr__max_leaf_nodes':[None, 1,2,3,4,5],
              }

grid = GridSearchCV(pipe, param_grid, scoring='r2')
grid.fit(X_train, Y_train)
print(f'Best estimators for ExtraTreesRegressor: {grid.best_estimator_}')
print(f'Best score is: {grid.best_score_}')

