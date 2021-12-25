import numpy as np
from matplotlib import pyplot as plt

def MSE(x,y, m, c):
    n = len(x)
    #print("n: ", n)
    yi = y
    #print("yi (actual): ,", yi)
    m = m #scaler
    #print("m (scaler): ", m)
    x = x #slope
    #print("x (slope): ", x)
    #print("this is c: ", c)
    c = c #intercept
    #print("c (intercept): ", c)
    y = -(m*x + c)
    pree = (yi + y)**2
    #print("(yi-y)^2: ", pree)
    e = sum(pree) / n
    #print("MSE: ", e)
    return e

def minimizeSlope(x,y,iterations):
    """Ended up using this method to find the best slope and intercept for machine learning.
    c = 0 = Starting Intercept, incriments by 1 +/- learning rate until it can find a slope that minimizes MSE
    a = 0 = Used to find the best slope for the equation, incriments +/- 1 * learning rate for each loop
    returns: w_history: history of different slopes tried before optimized slope was found, b_history: history of intercepts tried
    record: MSE's found through each iteration"""
    record = np.zeros(iterations)
    lrRecord = np.zeros(iterations)
    learningrate = .05
    c = 0
    a = 0
    b_history = [] #array of the history of change in intercept
    w_history = [] #array of the hitory of the change in slope

    for i in range(iterations):
        #print("iteration #: ", i)
        #print("current c: ", c)
        mse = MSE(x,y,(a+1)*learningrate, c)
        #print("mse: ", mse)
        nextMSE = MSE(x,y,(a-1)*learningrate, c)
        #print("nextMSE: ", nextMSE)
        if mse <= 0 or nextMSE <= 0:
            record[i] = mse
            lrRecord[i] = a * learningrate
            print("Exact line found. MSE = 0")
            return lrRecord, record, a * learningrate
        elif mse < nextMSE:
            c = minimizeIntercept(x, y, (a+1)*learningrate, c, mse, learningrate)
            #print("mse < nextMSE")
            record[i] = mse
            lrRecord[i] = a * learningrate
            b_history.append(c)  # history of change in intercept
            w_history.append(a*learningrate)  # history of the change in slope
            a = a + 1
        else:
            c = minimizeIntercept(x, y, (a-1)*learningrate, c, nextMSE, learningrate)
            #print("mse !< nextMSE")
            record[i] = nextMSE
            lrRecord[i] = a * learningrate
            b_history.append(c)  # history of change in intercept
            w_history.append(a*learningrate)
            a = a - 1
    return w_history,b_history, record

def minimizeIntercept(x,y,slope,c, mse, learningrate):
    """compares the mse with a line with an incrimented intercept and decremented intercept. Depending on which MSE is smaller it either incriments or decrements the slope"""
    intMSE_cup = MSE(x,y,slope,c+1+learningrate) #finds the cost (mse) if you were to increase c by 1 * learningrate.
    intMSE_cdown = MSE(x,y,slope,c-1+learningrate) #finds the cost (mse) if you were to decrease c by 1 * Learningrate
    #print("c up: ", intMSE_cup)
    #print("c same: ", mse)
    #print("c down: ", intMSE_cdown)
    if intMSE_cup > mse and intMSE_cdown > mse:
        return c
    if intMSE_cup <= mse:
        return c+1+learningrate
    elif intMSE_cdown <= mse:
        return c-1+learningrate

def graddescent(x,y,iterations):
    """Tried a Method to find the derivatives of slope and intercept in order to determine the weight. Ended up going another route"""
    learning_rate = 1
    slope = 0
    int = 3
    slope_history = np.zeros(iterations)
    int_history = np.zeros(iterations)
    #Requires arguments: x, y , theta?, learning rate, iterations
    # Performing Gradient Descent
    n = len(x)
    for i in range(iterations):
        #print("x in for loop: ", x)
        Y_pred = m * x + c  # The current predicted value of Y
        Der_slope = (-2 / n) * sum(x * (y - Y_pred))  # Derivative wrt m
        Der_int = (-2 / n) * sum(y - Y_pred)  # Derivative wrt c
        currentS = slope - learning_rate * Der_slope  # Update slope
        currentI = int - learning_rate * Der_int  # Update intercept
        slope_history[i] = currentS #add on to slope history
        int_history[i] = currentI #add on to intercept history
        #print("iteration #: ", i)
    #print(current_m, current_c)
    return slope_history, int_history



def gradient_descent(X, y, theta, learning_rate=0, iterations=10):
    # X = 0 #Matrix of X with added bias unit
    # y = 0 #Vector of Y
    # theta = 0 #Vector of thetas np.random.random(10) Z
    learning_rate = .1
    iterations = 5  # number of iterations

    # Returns the final theta vector and array of cost history over n number of iterations
    n = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / n) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)

    return theta, cost_history, theta_history
def cal_cost(theta,X,y):
    """
    Calculates the cost for given X and Y. The following shows an example of a single dimensional X. Wasn't really used in the end because i went about solving another way. Was found from: https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
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

def plotLine(slope,intercept,x_data,y_data):
    plt.scatter(x_data,y_data)
    x = np.linspace(0,60,100)
    y = slope*x+intercept
    label = "y = " + str(round(intercept, 4)) + " + " + str(round(slope, 4)) + "x"
    plt.plot(x, y, '-r', label=label)
    plt.title("Best Linear Line Found to Fit Data Set")
    plt.legend(loc="upper left")
    plt.show()

def plotLoss(calc_history,iterations):
    plt.plot(range(iterations),calc_history)
    plt.xlabel('Iteration #')
    plt.ylabel('MSE')
    plt.title("Loss Function")  # Slope = x, Intercept = y
    plt.show()

def LandscapeLossFunction(x_data, y_data):
    bb = np.arange(0, 100, 1)  # bias (intercept)
    #print("This is bb:", bb)
    ww = np.arange(-5, 5, 0.1)  # weight (slope)
    #print("This is ww:", ww)
    Z = np.zeros((len(bb), len(ww)))
    for i in range(len(bb)):
        for j in range(len(ww)):
            b = bb[i]
            w = ww[j]
            Z[j][i] = 0
            for n in range(len(x_data)):
                Z[j][i] = y_data[n] - (w * x_data[n] + b) #For Each combination of i/j value this represents the sum of distances to residuals for each line.
            Z[j][i] = (Z[j][i] * Z[j][i])/len(x_data) #makes all residuals positive
    #print("Length of Z: ", len(Z))
    #print("Length of x", len(x_data))
    #print("Z[1][1]: ", Z[1][1])
    return Z

def plotWeights(b_history,w_history, Z):

    if len(b_history) < len(Z):
        difference = len(Z) - len(b_history)
        for i in range(difference):
            b_history.append(b_history[-1])
            w_history.append(w_history[-1])
    plt.title("History of Slope and Intercept through Learning and Iterations")
    plt.plot(w_history,b_history, 'o-',ms=3,lw=1.5,color='black')
    plt.contourf(w_history, b_history,Z, 20, alpha=0.5, cmap=plt.get_cmap('jet'))
    plt.xlabel('b')
    plt.ylabel('w')
    #plt.contourf(b_history, w_history, Z, 20, alpha=0.5, cmap=plt.get_cmap('jet'))
    plt.show()


def main():
    x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
    y_data = 2 * x_data + 50 + 5 * np.random.random(10)  # updated

    iterations = 45

    slope, intercept, calc_history = minimizeSlope(x_data,y_data, iterations)

    print("Best Intercept found: ", intercept[-1])
    print("Best slope: ", slope[-1])
    Z = LandscapeLossFunction(x_data,y_data)

    plotLoss(calc_history,iterations)
    plotLine(slope[-1],intercept[-1], x_data,y_data)
    plotWeights(slope,intercept, Z)

    x_b = np.c_[np.ones((10, 1)), x_data]
    theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_data) #This is not my code, but was found from stack overflow (https://stackoverflow.com/questions/46586520/normal-equation-implementation-in-python-numpy/46590409)
    print("Theta best: ", theta_best)                                 #and was used to check my work


if __name__ == "__main__":
    main()


main()
