# -*- coding: utf-8 -*-
"""DropnaVal_v2_02

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19tkwZMh0dF6dLBdh5IDJzlq7YeYMmBIe
"""

!pip install fuzzywuzzy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process #package to correct misspelling labels
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

"""### Stuff

"""

df = pd.read_excel('mpg.data.xlsx')
df

df=df.drop(['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'], axis = 1)
df.columns = ['mpg',	'cylinders',	'displacements', 	'horsepower',	'weight',	'acceleration',	'model year',	'origin',	'car name']

df.isna().any()
df = df.dropna()

df.cylinders.astype('str')
df['model year'].astype('str')
df.origin.astype('str')

#df['displacements_by_cylinder'] = df.displacements/df.cylinders

df['brand_name'] = df['car name'].str.split(" ").str.get(0)

brand_list = ['chevrolet', 'mazda', 'mercedes-benz', 'toyota', 'volkswagen']

for brand in brand_list:

  matches = process.extract(brand, df['brand_name'], limit = df.shape[0])

  for potential_match in matches: 

    if potential_match[1]>70:

      df.loc[df['brand_name']==potential_match[0], 'brand_name']=brand


df.brand_name.replace('vw', 'volkswagen', inplace=True)

print(df.brand_name.unique())

"""### New Feature"""

df.cylinders.astype('str')
df['model year'].astype('str')
df.origin.astype('str')

"""df.loc[df['car name'].str.contains('diesel'), 'Diesel'] = 1
df.loc[~(df.Diesel==1), 'Diesel']=0"""

df.drop('car name', axis = 1 , inplace = True)
lb1=LabelEncoder()
lb2=LabelEncoder()
lb3=LabelEncoder()
lb4=LabelEncoder()

df['enc_origin'] = lb1.fit_transform(df.origin)
df['enc_cylinders'] = lb2.fit_transform(df.cylinders)
df['enc_brand'] = lb3.fit_transform(df.brand_name)
df['enc_year'] = lb4.fit_transform(df['model year'])

df.var()

"""###Filtering Outliers"""

df.loc[[8, 19, 102, 123]]

print(df[(df.horsepower>= df.horsepower.quantile(0.75) + 1.8*(df.horsepower.quantile(0.75)-df.horsepower.quantile(0.25)))])
df_no_outliers = df[~(df.horsepower>= df.horsepower.quantile(0.75) + 1.8*(df.horsepower.quantile(0.75)-df.horsepower.quantile(0.25)))]
df_no_outliers

print(df_no_outliers[(df_no_outliers.acceleration>= df_no_outliers.acceleration.quantile(0.75) + 2*(df_no_outliers.acceleration.quantile(0.75)-df_no_outliers.acceleration.quantile(0.25)))])

df_no_outliers=df_no_outliers[~(df_no_outliers.acceleration>= df_no_outliers.acceleration.quantile(0.75) + 2*(df_no_outliers.acceleration.quantile(0.75)-df_no_outliers.acceleration.quantile(0.25)))]

df_no_outliers

"""###Filtering"""

df_out = df.drop(['origin', 'model year','brand_name', 'cylinders', 'enc_brand', 'acceleration'], axis = 1)
df_no_out = df_no_outliers.drop(['origin', 'model year','brand_name', 'cylinders', 'enc_brand', 'acceleration'], axis = 1)
df_no_out.corr()

best_models_dict = {'ExtraTreesRegressor ':ExtraTreesRegressor(),  'RandomForestRegressor ': RandomForestRegressor(), 'GradientBoostingRegressor ': GradientBoostingRegressor(), 'BaggingRegressor':BaggingRegressor()}

scalers_dict = {'StandardScaler':StandardScaler() , 'MinMaxScaler':MinMaxScaler(), 'Normalizer':Normalizer(), 'MaxAbsScaler':MaxAbsScaler(),
                'RobustScaler':RobustScaler(), 'QuantileTransformer':QuantileTransformer(), 'PowerTransformer':PowerTransformer()}
"""models_dict ={'LinearRegression':LinearRegression(), 'Lasso':Lasso(), 'Ridge':Ridge(), 'DecisionTreeRegressor':DecisionTreeRegressor(), 'ExtraTreeRegressor':ExtraTreeRegressor(),  
         'BaggingRegressor':BaggingRegressor(), 'ExtraTreesRegressor':ExtraTreesRegressor(),
          'GradientBoostingRegressor':GradientBoostingRegressor(), 'RandomForestRegressor':RandomForestRegressor()} """

models_dict ={'DecisionTreeRegressor':DecisionTreeRegressor(), 'ExtraTreeRegressor':ExtraTreeRegressor(),  
         'BaggingRegressor':BaggingRegressor(), 'ExtraTreesRegressor':ExtraTreesRegressor(),
          'GradientBoostingRegressor':GradientBoostingRegressor(), 'RandomForestRegressor':RandomForestRegressor()}

### Find outliers with Local Outliers Factor

"""print(X_train.shape, y_train.shape)
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
print(X_train.shape, y_train.shape)
print(X2_train.shape, y2_train.shape)"""





X_train, X_test, y_train, y_test = train_test_split(df_out.drop('mpg', axis=1).values, df_out.mpg.values, test_size=0.2, random_state=21)
X2_train, X2_test, y2_train, y2_test = train_test_split(df_no_out.drop('mpg', axis=1), df_no_out.mpg, test_size=0.2, random_state=21)

"""###Function for cross_validation of a specific model and scaler pair"""



def cros_val(scaler_name, scaler, model_name, model, X, y, metric='neg_mean_squared_error'):
  pipe = Pipeline([(scaler_name, scaler), (model_name, model)])
  score = cross_val_score(pipe, X, y, cv=5, scoring=metric)
  print(f'The model is {model_name} with {scaler_name} as scaler and metrics scores {score} with mean {np.mean(score)}')
  return ((model_name, model), (scaler_name, scaler), score, np.mean(score))

"""list_models_mse = []
list_models_r2=[]
list_models_mse_no_outliers = []
list_models_r2_no_outliers = []

for scl_key, scl_val in scalers_dict.items():
  for mdl_key, mdl_val in models_dict.items():
    list_models_mse.append(cros_val(scl_key, scl_val, mdl_key, mdl_val, X_train, y_train))
    list_models_r2.append(cros_val(scl_key, scl_val, mdl_key, mdl_val, X_train, y_train, 'r2'))
    list_models_mse_no_outliers.append(cros_val(scl_key, scl_val, mdl_key, mdl_val, X2_train, y2_train))
    list_models_r2_no_outliers.append(cros_val(scl_key, scl_val, mdl_key, mdl_val, X2_train, y2_train, 'r2'))

### Top with outliers in the model

list_models_mse = sorted(list_models_mse, key=lambda x : x[3], reverse=True)
for i in range(10):
  print(f"{list_models_mse[i][0][0]} Model  and {list_models_mse[i][1][0]} with scores: {list_models_mse[i][2]} and Mean: {list_models_mse[i][3]:.3f}")
print('\n')
print('R2 \n')

list_models_r2 = sorted(list_models_r2, key=lambda x : x[3], reverse=True)
for i in range(10):
  print(f"{list_models_r2[i][0][0]} Model  and {list_models_r2[i][1][0]} with scores: {list_models_r2[i][2]} and Mean: {list_models_r2[i][3]*100:.3f}%")

### Top without outliers in the model

list_models_mse_no_outliers = sorted(list_models_mse_no_outliers, key=lambda x : x[3], reverse=True)
for i in range(10):
  print(f"{list_models_mse_no_outliers[i][0][0]} Model  and {list_models_mse_no_outliers[i][1][0]} with scores: {list_models_mse_no_outliers[i][2]} and Mean: {list_models_mse_no_outliers[i][3]:.3f}")

print('\n')
print('R2 \n')

list_models_r2_no_outliers = sorted(list_models_r2_no_outliers, key=lambda x : x[3], reverse=True)
for i in range(10):
  print(f"{list_models_r2_no_outliers[i][0][0]} Model  and {list_models_r2_no_outliers[i][1][0]} with scores: {list_models_r2_no_outliers[i][2]} and Mean: {list_models_r2_no_outliers[i][3]*100:.3f}%")

### Best Models
"""



def top_models(scaler_name, scaler, model_name, model, X, y, X_t, y_t):
  pipe = Pipeline([(scaler_name, scaler), (model_name, model)])
  pipe.fit(X, y)
  pred_test= pipe.predict(X_t)
  rmse = mean_squared_error(y_t, pred_test)**(1/2)
  mae= mean_absolute_error(y_t, pred_test)
  print(f'The model is {model_name} with {scaler_name} as scaler, RMSE: {rmse} , MAE: {mae} R2: {r2_score(y_t, pred_test)}')
  return ((model_name, model), (scaler_name, scaler), rmse, r2_score(y_t, pred_test), mae)

top_scores_no_outliers = []
top_scores_outliers = []
  
for scl_key, scl_val in scalers_dict.items():

  for name, model in best_models_dict.items():

    top_scores_outliers.append((top_models(scl_key, scl_val, name, model, X_train, y_train, X_test, y_test)))
    top_scores_no_outliers.append(top_models(scl_key, scl_val, name, model, X2_train, y2_train, X2_test, y2_test))

print('Models with outliers\n')
top_scores_outliers = sorted(top_scores_outliers, key=lambda x : x[2])
for i in range(8):
  print(f"{top_scores_outliers[i][0][0]} Model  and {top_scores_outliers[i][1][0]} with RMSE: {top_scores_outliers[i][2]} MAE: {top_scores_outliers[i][4]} and R2: {top_scores_outliers[i][3]*100:.3f}%")
print("\n")
print('Models without outliers\n')
top_scores_no_outliers = sorted(top_scores_no_outliers, key=lambda x : x[2])
for i in range(8):
  print(f"{top_scores_no_outliers[i][0][0]} Model and {top_scores_no_outliers[i][1][0]} with RMSE: {top_scores_no_outliers[i][2]} MAE: {top_scores_outliers[i][4]} and R2: {top_scores_no_outliers[i][3]*100:.3f}%")

scl = Normalizer()
X3_train = scl.fit_transform(X2_train)

temp = ExtraTreesRegressor()
score = cross_val_score(temp, X3_train, y2_train, cv=5, scoring= 'neg_mean_squared_error')
print(score)
rmse=(-1*(np.mean(score)))**(1/2)
print(rmse)

"""### HyperTunning
### 1) ExtraTreesRegressor  
### 2) RandomForestRegressor
### 3) GradientBoostingRegressor
### 4) BaggingRegressor

### 1) ExtraTreesRegressor
"""

'''etd2 = ExtraTreesRegressor(bootstrap=False,criterion='mse',max_depth=8, max_features='auto', min_samples_leaf=2,
 n_estimators= 100,  warm_start=True) '''
scl=Normalizer()
Xetd = scl.fit_transform(X2_train)
Xetd_t = scl.transform(X2_test)
etd2 = ExtraTreesRegressor(bootstrap=False ,criterion='mse', max_depth=25, max_features='auto', min_samples_leaf=1,
 n_estimators= 1500,  warm_start=True) 

etd2.get_params().keys()
etd2.fit(Xetd, y2_train)
prediction2  = etd2.predict(Xetd_t)
rmse2 = mean_squared_error(y2_test, prediction2)**(1/2)
print(rmse2)
print(r2_score(y2_test, prediction2))

etd2.get_params().keys()

"""### Random Forest"""

'''params_rfr = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

rfr = RandomForestRegressor()
gcv2 = RandomizedSearchCV(rfr, params_rfr, cv=5, random_state=42)
gcv2.fit(X2_train, y2_train)
prediction3  = gcv2.predict(X2_test)
rmse3 = mean_squared_error(y2_test, prediction3)**(1/2)
print(rmse3)

gcv2.best_params_ '''

scl_rf=  Normalizer()
X_rf = scl_rf.fit_transform(X2_train)
X_rf_t = scl_rf.fit_transform(X2_test)
rfr2 = RandomForestRegressor(bootstrap = True,
 max_depth=20,
 max_features= 'auto',
 min_samples_leaf= 1,
 min_samples_split=5,
 n_estimators=100)

rfr2.fit(X_rf, y2_train)
prediction4  = rfr2.predict(X_rf_t)
rmse4 = mean_squared_error(y2_test, prediction4)**(1/2)
print(rmse4)

"""###3) GradientBoostingRegressor

params_gbr = {'loss': ['ls', 'lad', 'huber'],
    'n_estimators': [100, 500, 900, 1100, 1500],
    'max_depth': [2, 3, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 10],
    'max_features': ['auto', 'sqrt', 'log2', None]}

gbr = GradientBoostingRegressor()
gcv3 = RandomizedSearchCV(gbr, params_gbr, cv=5, verbose = 5,            
            random_state=21, scoring = 'neg_mean_squared_error')
gcv3.fit(X2_train, y2_train)
prediction5  = gcv3.predict(X2_test)
rmse3 = mean_squared_error(y2_test, prediction5)**(1/2)
print(rmse3)

gcv3.best_params_"
"""

scl_gbr=MinMaxScaler()
X_gbr= scl_gbr.fit_transform(X2_train)
X_t_gbr = scl_gbr.transform(X2_test)
gbr = GradientBoostingRegressor()
gbr.fit(X_gbr, y2_train)
prediction5  = gbr.predict(X_t_gbr)
rmse3 = mean_squared_error(y2_test, prediction5)**(1/2)
print(rmse3)

2.126697571533698

"""Testing"""

temp_scl= Normalizer()
X_cross = temp_scl.fit_transform(df_no_out.drop('mpg', axis=1).values)
score = cross_val_score(ExtraTreesRegressor(), X_cross, df_no_out['mpg'].values, cv=3, scoring = 'neg_mean_squared_error')
print(np.mean(score))
rmse = (-1*np.mean(score))**0.5
print(rmse)