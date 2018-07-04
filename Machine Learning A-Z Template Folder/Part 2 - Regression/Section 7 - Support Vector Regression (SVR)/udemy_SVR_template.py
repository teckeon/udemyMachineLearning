#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:27:58 2018

@author: teckeon
"""


'''
SVR Template
'''

# Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # want to see x as a matrix and vectorso we add 2
y = dataset.iloc[:, 2:3].values # originally [:,2] but was getting a straight line and this fixed it


# Splitting the dataset into the Training set and Test set
# not enough data to train due to the small sample this dataset only has 10
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) '''

# Feature Scaling
# in this example we are feature scaling since importing SVR does not have 
# this feature so we must include it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR to the dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# Predicting a new result 

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) # Need to tranfrom 6.5 and then inverse it back from when it was scaled





# Visualizing the  SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Replace x label text here')
plt.ylabel('Replace y label text here')
plt.show()


# Visualizing the  SVR results (for higher resolution and smoother curv)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Replace x label text here')
plt.ylabel('Replace y label text here')
plt.show()
