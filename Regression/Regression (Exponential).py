# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:28:36 2020

@author: Jacob Watters

Use this program to fit single variable exponential functions of the form (ax^b where a and b are reals) to data.
If you are going to fit an exponential model to data as a method of analysing the data, you should have at least some 
reason to believe that the source of the data would behave exponentially. 
"""

import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import math


# S is set of (x, y) points 
def least_squares_pow(S):
    # Extract x and y values from S
    x = [S[i, 0] for i in range(len(S))]
    y = np.log([S[i, 1] for i in range(len(S))])
    
    V = gen_vandermonde(x)
    VT = transpose(V)
    A = np.dot(VT, V)
    b = np.dot(VT, y)
    
    # Coeficients of least squares polynomial
    c = solve(A, b)
    
    return c


# Generates the partial "vandermonde-like" matrix given a vector x
def gen_vandermonde(x):
    V = np.ones((len(x), 2))
    
    for i in range(len(x)):
        V[i, 1] = np.log(x[i])
            
    return V


# Transposes Matrix A
def transpose(A):
    m, n = A.shape
    AT = np.zeros((n, m))
    
    for i in range(m):
        for j in range(n):
            AT[j, i] = A[i, j]
            
    return AT


# c is polynomial coefficents, interval is a tupple defining range to compute values, n is number of sample values 
# n must be greater than or equal to 2
def pow_vals(c, interval=(-1, 1), n=100):
    vals = np.zeros((n, 2))
    a, b = interval
    
    for i in range(n):
        x_val = a + i*(b-a)/(n-1)
        y_val = (math.e**c[0])*((x_val)**c[1])
            
        vals[i, 0] = x_val
        vals[i, 1] = y_val
        
    return vals
    
      
def main():
    
    # Set of (x, y) points
    S = np.array(  [[10, 94],
                    [16, 118],
                    [25, 147],
                    [40, 180],
                    [60, 230],], float)
    
    
    c = least_squares_pow(S)
    
    interval = [min(S[:,0]), max(S[:,0])]
    powVals = pow_vals(c, interval, n=100)
    x_vals = powVals[:, 0]
    y_vals = powVals[:, 1]
    
    # Plot Data points
    plt.plot(S[:,0], S[:,1], 'ro', label="data")
    # Plot line
    plt.plot(x_vals, y_vals, label="least squares")
    plt.title("Least Squares (Power Law)")
    plt.legend(loc="upper left")
    plt.show()
   
    print("Values:  ", c)
    print("Equation: y = %.4fx^%.4f" %(math.e**c[0], c[1]))

    
if __name__ == "__main__":
    main()
