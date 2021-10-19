#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:15:05 2021

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

from lightgbm import LGBMRegressor as lgbreg
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.ensemble import ExtraTreesRegressor, IsolationForest

path_train = '/home/mav24/Documents/Development/Regeneration/Project/Data/training_data.xlsx'
data_train = pd.read_excel(path_train)
drop = ['Unnamed: 0', 'station wagon', 'encoded origin', 'diesel']
data_train.drop(columns=drop, inplace=True)

path_val = '/home/mav24/Documents/Development/Regeneration/Project/Data/validation_data.xlsx'
data_val = pd.read_excel(path_val)

data_val.drop(columns=drop, inplace=True)




# Scaling the data Standar sceler
X_train = data_train.drop(columns='mpg')
Y_train = data_train['mpg']
X_test = data_val.drop(columns='mpg')
Y_test = data_val['mpg']
scaler = StandardScaler()

# Sanity check of shapes
print ('Shape of X_train=>',X_train.shape)
print ('Shape of X_test=>',X_test.shape)
print ('Shape of Y_train=>',Y_train.shape)
print ('Shape of Y_test=>',Y_test.shape)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Outliers Detection
iso = IsolationForest(contamination=0.05)
yhat = iso.fit_predict(X_train)

mask = yhat != -1
X_train, Y_train = X_train[mask, :], Y_train[mask]



model = lgbreg()
cros = np.mean(cross_val_score(model, X_train, Y_train, cv=10, scoring='r2'))
print(f'The mse from cross val: {np.sqrt(cros)}')

model.fit(X_train, Y_train)
pred = model.predict(X_test)
rmse = np.sqrt(mse(Y_test, pred))
print(f'Prediction mse: {np.sqrt(cros)}')
print(f'Prediction r2: {r2(Y_test, pred)}')
