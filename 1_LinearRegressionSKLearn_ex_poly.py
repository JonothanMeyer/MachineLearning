#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the coefficient
of determination are also calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Load the diabetes dataset
diabetes_X_old, diabetes_y = datasets.load_diabetes(return_X_y=True)

## Use only one feature
diabetes_X = diabetes_X_old[:,np.newaxis, 2]
#
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-60]
diabetes_X_test = diabetes_X[-60:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-60]
diabetes_y_test = diabetes_y[-60:]

# Create linear regression object
regr = LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

regr2 = LinearRegression()
poly2_features = PolynomialFeatures(degree=10)
X_poly2 = poly2_features.fit_transform(diabetes_X_train)
X_poly2_test = poly2_features.fit_transform(diabetes_X_test)
regr2.fit(X_poly2, diabetes_y_train)
diabetes_y_pred2 = regr2.predict(X_poly2_test)



# The coefficients
#print('Coefficients: \n', regr.coef_)
#print('Itercept: \n', regr.intercept_)
# The mean squared error
print('Mean squared error of linear model: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Mean squared error of poly2 model: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred2))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))
print('Coefficient of determination ploy2: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred2))


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.plot(diabetes_X_test, diabetes_y_pred2, color='red', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
