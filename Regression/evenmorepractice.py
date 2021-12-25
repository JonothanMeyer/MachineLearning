import numpy as np
from matplotlib import pyplot as plt

def gradient_descent(X,y,theta,learning_rate =0.01,iterations=20):
    """
    X = Matrix of X with added bias units
    y = Vector of Y
    theta = Vector of thetas np.random.rand(j,1)
    learning_rate
    iterations = no of iterations
    Returns the final theta vector and array of cost history over no of iterations
    """
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    print("initial theta history: ", theta_history)
    for it in range(iterations):
        prediction = np.dot(X,theta)
        theta = theta -(1/m)*learning_rate*(X.T.dot((prediction - y)))
        theta_history[it,:] = theta.T

        cost_history[it] = cal_cost(theta,X,y)

    return theta, cost_history, theta_history

def cal_cost(theta,X,y):
    """
    Calculates the cost for given X and Y. The following shows an example of a single dimensional X
    theta = Vector of thetas
    X = Row of X's np.zeros((2,j))
    y = Actual y's np.zeros((2,1))
    where:
        j is the no of features
        """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    print("cost: ", cost)
    return cost

def plotCost(iterations, cost_history):
    fig,ax = plt.subplots(figsize=(12,8))

    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    _=ax.plot(range(iterations),cost_history,'b.')
    plt.show()


def main():
    #x_data = np.array([35, 38, 31, 20, 22, 25, 17, 60, 8, 60], float)
    x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
    y_data = 2 * x_data + 50 + 5 * np.random.random(10)  # updated
    print("y data: ", y_data)
    #y_data = 4 + 3 * x_data + np.random.randn(100, 1)
    #x_data = 2 * x_data + 50 + 5*np.random.randn(10,1)


    plt.scatter(x_data,y_data)
    plt.show()

    learning_rate = 0.05
    iterations = 20
    #theta = np.random.random(2)
    print("trying different things: ", np.random.rand())
    theta = np.array([0, 0])
    print("Theta: ", theta)

    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((10, 1)), x_data]

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_data)
    print("Theta best: ", theta_best)
    print("x_b: ", X_b)
    print("x: ", x_data)
    theta, cost_history,theta_history = gradient_descent(X_b, y_data, theta, learning_rate, iterations)
    print('Theta0 (intersect/B): ', theta[0])
    print('Theta1 (slope/m): ', theta[1])
    print("Final cost/MSE: ", cost_history[-1])


    plotCost(iterations, cost_history)

main()