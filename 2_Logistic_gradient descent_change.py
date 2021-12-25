# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
 
def sigmoid_activation(z):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-z))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.0001,
	help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 250 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
	cluster_std=1.05, random_state=20)
 
# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]
 
# initialize our weight matrix such it has the same number of
# columns as our input features
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))
 
# initialize a list to store the loss value for each epoch
lossHistory = []
lossHistoryCE = []
# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid_activation(X.dot(W))
    error = preds - y   
    loss = np.sum(error ** 2) / X.shape[0]
    lossHistory.append(loss)
    print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))
    num_samples = X.shape[0]
    loss_CE =0
    for index in range(num_samples):
        loss_CE = loss_CE-y[index]*np.log(preds[index])-(1-y[index])*np.log(1-preds[index])  	
    print("[INFO] epoch #{}, cross_entropy loss={:.7f}".format(epoch + 1, loss_CE))
    lossHistoryCE.append(loss_CE)
    
    gradient = X.T.dot(error) #/ X.shape[0]
    W += -args["alpha"] * gradient
    
    

# to demonstrate how to use our weight matrix as a classifier,
# let's look over our a sample of training examples
for i in np.random.choice(250, 10):
	# compute the prediction by taking the dot product of the
	# current feature vector with the weight matrix W, then
	# passing it through the sigmoid activation function
	activation = sigmoid_activation(X[i].dot(W))
 
	# the sigmoid function is defined over the range y=[0, 1],
	# so we can use 0.5 as our threshold -- if `activation` is
	# below 0.5, it's class `0`; otherwise it's class `1`
	label = 0 if activation < 0.5 else 1
 
	# show our output classification
	print("activation={:.4f}; predicted_label={}, true_label={}".format(
		activation, label, y[i]))
    

plotx =X[:, 1]
ploty =(-W[0] - (W[1] * plotx)) / W[2]
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(plotx, ploty, "r-")
# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistoryCE,color='red')
fig.suptitle("Training Loss CrossEntropy")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory,color='blue')
fig.suptitle("Training Loss MSE")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()