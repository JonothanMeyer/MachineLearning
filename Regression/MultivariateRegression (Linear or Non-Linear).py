# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:04:19 2020

@author: Jacob Watters

Use this program to preform linear or nonlinear regression to fit mulivariate functions to data.
Note: If you change the function r, you must recompute the Jacobian Matrix (Dr) by hand 
(or some calculator tool) and modify it here.
"""

import numpy as np
import matplotlib.pyplot as plt


# An example vector-valued function y = r(x_1, x_2, ..., x_n) with n independent valriables and constant values c (Could be any funciton)
def r(c, x, y): 
    A = c[0]
    B = c[1]
    
    return np.array([A / (_x + B) - _y for _x, _y in zip(x, y)])


# An example Jacobian of r(c, x, y) (Would be diffrent for diffrent r function)
def Dr(c, x, y):
    A = c[0]
    B = c[1]
    
    return np.array([[1 / (_x + B), -A / (_x + B)**2] for _x in x])


# Solves Non-linear least squares via Gauss Newton method
def gauss_newton_method(c, x, y, tol=0.5e-08, maxiters=100):
    iters = 0
    while True:
        A = Dr(c, x, y)
        
        # could be done with LU, QR, etc.
        v = np.linalg.solve(A.T @ A, -A.T @ r(c, x, y)) # @ is matrix multiplication (outer product)
        c_new = c+v
        iters += 1
        
        # Check infinity norm of risidual
        if np.max(np.abs(c_new - c)) > tol and iters < maxiters:
            c = c_new # then update c
        else:
            return c_new


# Returns error of gauss newton method
def GN_error(c, x, y):
    return np.dot(r(c, x, y), r(c, x, y))
    

# Linear Least Squares (For comparison of gauss_newton_method. Can only do this if r is linear/linearizable)
def Linear_LS(x, y):
    V = np.array([np.ones(len(x)), -y]).T
    b = np.array(x*y)
    c = np.linalg.solve(V.T @ V, V.T @ b)
    return c


def Linear_LS_error(c, x, y):
    V = np.array([np.ones(len(x)), -y]).T
    b = np.array(x*y)
    residual = b - V @ c
    return np.dot(residual, residual) # Sum of square errors
    

def main():
    x = np.linspace(1.0, 5.0, 5)
    y = np.array([2.0, 5.0, 10.0, 17.0, 26.0])
    
    c1 = gauss_newton_method(np.array([-17.0, -6.0]), x, y)
    c2 = Linear_LS(x, y)
        
    # Keep in mine the SSE's are in a linear space, and a non-linear(in this case logarithmic) space. 
    print()
    print("{S:^28s}".format(S = "Coeffs (A, B)"), end = '|')
    print("{S:^14s}".format(S = "Nonlinear SSE"), end = '|')
    print("{S:^14s}".format(S = "Linear SSE"))
    print(54*"-")
    for c in [c1, c2]:
        print("({A:12.8f}, {B:12.8f})".format(A=c[0], B=c[1]), end = "|")
        print("{E:14.4f}".format(E=GN_error(c, x, y)), end = "|")
        print("{E:12.4f}".format(E=Linear_LS_error(c, x, y)))
    print()
    
    
    # PLease note this is only graphing the x-y plane (dependent variables) not the full space of the function.
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    ax.scatter(x, y, color='black', alpha=0.5, edgecolor = 'none', label = "data")
    t = np.linspace(1.0, 5.0, 201)
    A = c1[0]
    B = c1[1]
    ax.plot(t, [A / (_t + B) for _t in t], label = "nonlinear")
    A = c2[0]
    B = c2[1]
    ax.plot(t, [A / (_t + B) for _t in t], label = "linearized")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Nonlinear vs Linear Least Squares")
    ax.legend()
    plt.show()
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    

