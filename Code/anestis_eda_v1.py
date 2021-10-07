# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 22:25:13 2021

@author: Maverick
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Functions
"""

def check_scat_for_connections(lista):
    for i in lista:
        for j in lista:
            sns.regplot(x=data[i], y=data[j], data=data)
            plt.show()
            
def check_outliers_for_connections(lista):
    for i in lista:
        for j in lista:
            sns.boxplot(x=data[i], y=data[j], data=data)
            plt.show()


def distribution_plot (x):
    #nbins = int(x.max())-int(x.min())+1
    #print (type(nbins))
    ax = sns.distplot(x, )
    
    # Auxiliary information:
    mn = x.mean()
    mx = ax.lines[0].get_ydata().max()
    
    # Plot median line:
    ax.plot([mn]*2, [0, mx])
    
    
    sns.set(style='white', palette='muted', color_codes=True)
    plt.annotate('Mean', [mn, mx], xytext=[mn*1.1, mx*1.1], fontsize=10, arrowprops=dict(arrowstyle='->', color='red',connectionstyle='arc3, rad=.2'))
    plt.show()


def correction_brands ( list_of_brands):
    """
    Correcting the name of the brands that are misspelt
    Input:
        list_of_brands : List of car brands
    
    Output:
        corr_list : List with the names of the brands corrected
    """
    dict_of_corrections = {'chevroelt':'chevrolet', 'chevy':'chevrolet', 'maxda':'mazda', 'vokswagen':'vw'}
    corr_list = list_of_brands.replace(dict_of_corrections)
    return corr_list


"""
Reading the Data
"""

data=pd.read_excel('/home/mav24/Documents/Development/Regeneration/Project/Data/mpg.data.xlsx')

"""
EDA
"""

# Droping Empty columns
data.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'], inplace=True)
data.columns=['mpg', 'cylinders', 'displacements', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

# Sanity check. Count of all the data (rows)
print (f'Total number of data: {len(data)}')

# Dropping rows with na 
data.dropna(axis=0, inplace=True)
print (f'Total number of data after dropping NaN values: {len(data)}')

# Explanation of what data mean
print ('MPG: Miles-per-Gallon (consuption)')
print ('Cylinders: Number of cylinders in the car')
print ('Displacement: The volume of cc in every cylinder')
print ('Horsepower: The measurement of the power of the engine')
print ('Weight: Weight of the car in lbs')
print ('Acceleration: Total time , in seconds,taken to reach 100km/h')
print ('Model year: Yeah of the car assemply')
print ('Origin: The continent the car was made')
print ('Car Name: Brand and model of the car')


"""
Some preperation on the data
"""

# Seperate the car name to brand and correcting it 
car_brand = data['car name'].str.split().str.get(0)
car_brand = correction_brands(car_brand)
data['car brand'] = car_brand


# All numeric columns
numeric_columns = ['mpg', 'cylinders', 'displacements', 'horsepower', 'weight', 'acceleration']


# Get statistical values of numerical columns
stat_data = data.loc[:, numeric_columns].describe()


# Check scatter graphs for connections on numerical columns
check_scat_for_connections(numeric_columns)


"""
Scatter Plots
"""


# MPG Diplacements polynomial model 2nd order
sns.regplot(x=data['displacements'], y=data['mpg'], data=data, order=2, line_kws={'color':'red'})
plt.xlabel('Displacements')
plt.ylabel('MPG')
plt.title('')
plt.show()
print ('Second order polynomial model fit the correlation between MPG and Displacements')


# Horsepower displacements linear model 
sns.regplot(x=data['horsepower'], y=data['displacements'], data=data, line_kws={'color':'red'})
plt.xlabel('Horsepower')
plt.ylabel('Displacements')
plt.show()
print ('Linear model fit the correlation between MPG and Displacements')


"""
Barplots
"""


# Barplot for brand names
unique_manufac = data['car brand'].unique()
plt.bar(x=unique_manufac, height=pd.value_counts(data['car brand']))
plt.xticks(rotation=90)
plt.title('Car manufactures')
plt.ylabel('Number of cars')
plt.show()


# Barplot for the continets that the cars are made
cont=['America', 'Europe', 'Asia']
unique_origin = data['origin'].unique()
plt.bar(x=cont, height=pd.value_counts(data['origin']))
plt.title('Continents of car manufactures')
plt.ylabel('Number of cars')
plt.show()



# Distribution plots 
distribution_plot(data['mpg'])
distribution_plot(data['displacements'])
distribution_plot(data['horsepower'])
distribution_plot(data['weight'])
distribution_plot(data['acceleration'])


# Seaborn barplot
sns.barplot(x=data['cylinders'], y=data['displacements'], data=data)

plt.show()

# Seaborn Boxplot
sns.boxplot(x=data['cylinders'], y=data['horsepower'], data=data)
plt.show()

# Seaborn Boxplot
sns.boxplot(x=data['model year'], y=data['mpg'], data=data)
plt.show()

sns.barplot(x=data['model year'], y=data['mpg'], data=data)
plt.show()

check_outliers_for_connections(numeric_columns)
