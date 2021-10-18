#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:36:38 2021

@author: mav24
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer, StandardScaler, PowerTransformer, MaxAbsScaler

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split


"""
Reading the training data
"""
path = '/home/mav24/Documents/Development/Regeneration/Project/Data/training_data.xlsx'
data = pd.read_excel(path)

#data.drop(columns=['Unnamed: 0', 'diesel', 'station wagon'], inplace=True)
drop = ['Unnamed: 0', 'encoded car brand', 'station wagon', 'cylinders', 'encoded origin']
data.drop(columns=drop, inplace=True)


# Scaling the data Standar sceler
X = data.drop(columns='mpg')
Y = data['mpg']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
# Outliers Detection
iso = IsolationForest(contamination=0.05)
yhat = iso.fit_predict(X_scaled)

mask = yhat != -1
X_scaled, Y = X_scaled[mask, :], Y[mask]
"""

# Splitting the training data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=13)



# Training and prediction
model = ExtraTreesRegressor()
#model = GradientBoostingRegressor()
model.fit(X_train, Y_train)
pred_test = model.predict(X_test)

print('With Standar Scaler')
print(f'The R2 accuracy is: {r2(Y_test, pred_test)}')
print(f'The mean square error is: {mse(Y_test, pred_test)}')
print(f'Mean absolute error is: {mae(Y_test, pred_test)}')





model_for_cross = ExtraTreesRegressor()
#model_for_cross = GradientBoostingRegressor()
cross_val = cross_val_score(model_for_cross, X_scaled, Y, cv=10, scoring='neg_root_mean_squared_error')
print(f'Cross validation is: {cross_val} \n and mean: {np.mean(cross_val)} \n and std:{np.std(cross_val)}')




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
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
"""


"""

# Scaling the data PowerTransformer
X = data.drop(columns='mpg')
Y = data['mpg']
scaler = PowerTransformer()
X_scaled = scaler.fit_transform(X)


# Splitting the training data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=13)



# Training and prediction
model = ExtraTreesRegressor()
model.fit(X_train, Y_train)
pred_test = model.predict(X_test)


print('With PowerTransformer')
print(f'The R2 accuracy is: {r2(Y_test, pred_test)}')
print(f'The mean square error is: {mse(Y_test, pred_test)}')
print(f'Mean absolute error is: {mae(Y_test, pred_test)}')

"""

"""
Validate the model to unseen data
"""

#path_val = '/home/mav24/Documents/Development/Regeneration/Project/Data/vavlidation_data.xlsx'
#data_val = pd.read_excel(path_val)





