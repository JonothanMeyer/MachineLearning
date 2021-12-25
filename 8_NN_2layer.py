#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Oct 2, 2018
@author: Madhuri Suthar, UCLA
"""
# Imports

import numpy as np
import matplotlib.pyplot as plt

 # Each row is a training example, each column is a feature  [X1, X2, X3]
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)
#y=np.array(([0,1],[1,0],[1,0],[0,1]), dtype=float)

print("Shape of x: ", X.shape) # X: (4, 3) 4 row, 3 columns = weight matrix 1
print("Shape of y: ", y.shape) # y: (4, 1) = weight matrix 2

# Define useful functions

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
def plot_loss(loss_history, iteration_history):
    #y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    #fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    #auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(iteration_history, loss_history, label="Final Loss = " + str(loss_history[-1]))
    plt.legend(loc=1)
    plt.title('Loss History Through Iterations')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()

class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4) # considering we have 4 nodes in the hidden layer
        print("weights1 shape: ", self.weights1.shape)
        self.weights2 = np.random.rand(4,1)
        print("weights2 shape: ", self.weights2.shape)
        self.y = y
        self.output = np. zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*
                                                 sigmoid_derivative(self.output), self.weights2.T)*
                                                 sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()

NN = NeuralNetwork(X,y)
Loss_history = []
iteration_history = []

for i in range(1500): # trains the NN 1,000 times
    if i % 300 ==0:
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        loss = np.mean(np.square(y - NN.feedforward()))
        iteration_history.append(i)
        Loss_history.append(loss)
        print ("Loss: \n" + str(loss)) # mean sum squared loss
        print ("\n")
    NN.train(X, y)

plot_loss(Loss_history, iteration_history)