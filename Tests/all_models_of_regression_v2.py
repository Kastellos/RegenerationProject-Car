#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:53:59 2021

@author: mav24
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder


"""
Scalers
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


"""
Regression Models
"""
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.svm import SVR


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split

path = '/home/mav24/Documents/Development/Regeneration/Project/Data/training_data.xlsx'
data = pd.read_excel(path)


list_of_models = [LinearRegression(), SGDRegressor(), KNeighborsRegressor(),
                          GaussianProcessRegressor(), PLSRegression(), 
                          DecisionTreeRegressor(), ExtraTreeRegressor(), 
                          AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(), RandomForestRegressor(),
                          SVR()]


scalers = [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer()]

data.drop(columns='Unnamed: 0', inplace=True)
cols = data.columns.tolist()




acc = 0
mse = 0
mae = 0
for i in list_of_models:
    
    for j in scalers:
        for k in cols:
            if k !='mpg':
                print(k)
                
                
                data_testing = data.drop(columns=k)
                print (data_testing.columns)
                
                X = data_testing.drop(columns='mpg')
                Y = data_testing['mpg']
                
                scaler = j
                X_scaled = j.fit_transform(X)
                
                X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=13)
                
                
                model = i
                model.fit(X_train, Y_train)
                pred_test = model.predict(X_test)
                print(f'Model name is: {i}')
                print(f'Scaler: {j}')
                print(f'Attribute dropped: {k}')
                print(f'The R2 accuracy is: {r2_score(Y_test, pred_test)}')
                print(f'The mean square error is: {mean_squared_error(Y_test, pred_test)}')
                print(f'Mean absolute error is: {mean_absolute_error(Y_test, pred_test)}\n\n')
            r2 = r2_score(Y_test, pred_test)
            if r2>acc:
                acc = r2
                best_model = i
                best_scaler = j
                best_drop = k
                r2_main = r2
                mse = mean_squared_error(Y_test, pred_test)
                mae = mean_absolute_error(Y_test, pred_test)
            print (f'Best model so far: {best_model}, best scaler: {best_scaler} and best dropped attribute: {best_drop}')
            print (f'It has R2: {r2_main}, MSE: {mse} and MAE: {mae}\n')


