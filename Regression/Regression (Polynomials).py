# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:28:36 2020

@author: Jacob Watters

Use this program to fit single variable polynomials of various degrees (degree 1 = linear regression) onto data.
"""

import numpy as np
import matplotlib.pyplot as plt


# Solves the linear systems Ax=b. Returns x
def solve(A, b):
   L, U = LU_factor(A)

   y = forward_solve(L, b)
   x = back_solve(U, y)
   
   return x

# Decomposes A into a unit-lower triangular matrix L and an upper triangular matrix U. That is A=LU
def LU_factor(A):
    n = len(A)
    LU = np.copy(A)
    I = identity(n)
   
    # Using Gaussian elimination (no pivoting)... 
    for j in range(n):     
        for i in range(j+1, n, 1):
            m = LU[i, j] / LU[j, j]
            for k in range(j, n, 1):
                LU[i, k] = LU[i, k] - m*LU[j, k]
                LU[i, j] = m
    
    
    L = np.tril(LU, -1) + I
    U = np.triu(LU)
    
    return L, U


# Solve Ly=b where L is a nxn unit-lower triagular matrix and b is a vector of size n.
def forward_solve(L, b):
    n = len(b)
    y = np.zeros(n)
    
    y[0] = b[0] 
    
    for i in range(n):
        ysum = 0
        for j in range(i-1, -1, -1):
            ysum += L[i][j]*y[j]
        y[i] = (b[i] - ysum)
        
    return y


# Solve Ux=y in O(n^2) where U is a nxn upper triangular matrix and y is a vector of size n
def back_solve(U, y):
    n = len(y)
    x = np.zeros(n)
    
    x[n-1] = y[n-1]/U[n-1][n-1] 
    
    for i in range(n-1, -1, -1):
        xsum = 0
        for j in range(i+1, n):
            xsum += U[i][j]*x[j]
        x[i] = (y[i] - xsum)/U[i][i]
        
    return x


# Returns the identity matrix with given size
def identity(size):
    I = np.zeros((size, size))
    
    for i in range(size):
        I[i, i] = 1.0
        
    return I


# S is set of (x, y) points and n is the degree of the polynomial to be fit onto data
def least_squares_poly(S, n=1):
    # Extract x and y values from S
    x = [S[i, 0] for i in range(len(S))]
    y = [S[i, 1] for i in range(len(S))]
    
    V = gen_vandermonde(x, n+1)
    VT = transpose(V)
    
    # Matrix Normal Equation: VT*Vx=VTy
    A = np.dot(VT, V)
    b = np.dot(VT, y)
    
    # Solve the linear System Ac = b. c is coeficients of least squares polynomial
    c = solve(A, b)
    
    return c


# Generates the vandermonde matrix given a vector of parameters x and a count n. returens len(x) x n vandermonde matrix
def gen_vandermonde(x, n):
    V = np.ones((len(x), n))
    
    for i in range(len(x)):
        for j in range(1, n):
            V[i, j] = x[i]**j
    
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
def poly_vals(c, interval=(-1, 1), n=100):
    vals = np.zeros((n, 2))
    a, b = interval
    
    for i in range(n):
        x_val = a + i*(b-a)/(n-1)
        y_val = 0
        for j in range(len(c)):
            y_val += c[j]*((x_val)**j)
            
        vals[i, 0] = x_val
        vals[i, 1] = y_val
        
    return vals


# Evaluated polynomial with coefficents c at given x value
def eval_poly(c, x):
    y_val = 0
    
    for j in range(len(c)):
        y_val += c[j]*((x)**j)
        
    return y_val
            

# c is polynomial coeficents, S is set of (x, y) points
def variance(c, S):
    N = len(S)
    n = len(c)-1
    
    errorSquareSum = 0
    
    for i in range(N):
        errorSquareSum += abs(eval_poly(c, S[i, 0])-S[i, 1])
    
    var = (errorSquareSum) / (N-n-1)
    
    return var


# c is a list of coefficents (starting with constant term, to highest degree term)
# c must be of length 2 or more. a is the number of decimal places to print
def print_poly(c, a=2):
    poly = 'y = ' + str(round(abs(c[0]), a)) + (' - ' if c[1] < 0 else ' + ') + str(round(abs(c[1]), a)) + 'x'
    
    if len(c) > 2:
        poly+=(' - ' if c[2] < 0 else ' + ')
    
    for i in range(2, len(c)):
        poly += str(round(abs(c[i]), a)) + 'x^' + str(i)
        if i < len(c)-1:
            poly+=(' - ' if c[i] < 0 else ' + ')
            
    print(poly)
      

def main():
    
    #For testing...
    
    # Some sample data (year/year % ave income change by % Incumbent votes)
    S = np.array(  [[1.49, 44.6],
                    [3.03, 57.8],
                    [0.57, 49.9],
                    [5.74, 61.3],
                    [3.51, 49.6],
                    [3.73, 61.8],
                    [2.98, 49.0],
                    [-0.18, 44.7],
                    [6.23, 59.2],
                    [3.38, 53.9],
                    [2.15, 46.5],
                    [2.10, 54.7],
                    [3.93, 50.3],
                    [2.47, 51.2],
                    [-0.41, 45.7]], float)
    
    ######################## Fit polynpomial of degrees 1 S data ########################
    c = least_squares_poly(S)
    var = variance(c, S)
    
    interval = [min(S[:,0]), max(S[:,0])]
    polyVals = poly_vals(c, interval, n=100)
    x_vals = polyVals[:, 0]
    y_vals = polyVals[:, 1]
    
    # Plot Data points
    plt.plot(S[:,0], S[:,1], 'ro', label="data")
    # Plot line
    plt.plot(x_vals, y_vals, label="least squares")
    plt.title("% Incumbent Vote vs % income change")
    plt.legend(loc="upper left")
    plt.xlabel('Year/Year % Income Change')
    plt.ylabel('% Incumbent Vote')
    plt.show()
   
    print("Polynomial Coefficents: ", c, "\nEquation: ", end="")
    print_poly(c)
    print("Varience: %.6f" %var)
    ############################################################################################
    
    S1 = np.array([[0.05, 0.956],
                    [0.11, 0.890],
                    [0.15, 0.832],
                    [0.31, 0.717],
                    [0.46, 0.571], 
                    [0.52, 0.539],
                    [0.70, 0.378], 
                    [0.74, 0.370],
                    [0.82, 0.306], 
                    [0.98, 0.242],
                    [1.171, 0.104]], float)
    
    interval = [min(S1[:,0]), max(S1[:,0])]
    
    ######################## Fit polynpomial of degrees 1-9 on S1 data ########################
    for i in range(1, 10):
        # Plot Data points
        plt.plot(S1[:,0], S1[:,1], 'ro', label="data")
        c = least_squares_poly(S1, i)
        var = variance(c, S1)
        polyVals = poly_vals(c, interval, n=100)
        x_vals = polyVals[:, 0]
        y_vals = polyVals[:, 1]
        plt.plot(x_vals, y_vals, label="least squares")
        plt.legend(loc="upper right")
        plt.title("Least Squares (Degree " + str(i) + " Polynomial)")
        plt.show()
        
        print("Polynomial Coefficents: ", c, "\nEquation: ", end="")
        print_poly(c)
        print("Variance: %.6f" %var)
    ############################################################################################
    
    
if __name__ == "__main__":
    main()
