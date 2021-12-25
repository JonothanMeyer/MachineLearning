import numpy as np
from matplotlib import pyplot as plt


def residual(x, y, i):
    return y[i] - pred(x, y, x[i])


def mean(data):
    n = len(data)
    ave = 0
    for i in range(n):
        ave += data[i]
    return ave / n


def S(x, y):
    x_bar = mean(x)
    y_bar = mean(y)

    n = len(x)
    result = 0
    for i in range(n):
        result += (x[i] - x_bar) * (y[i] - y_bar)

    return result


def r(x, y):
    result = S(x, y) / (np.sqrt(S(x, x)) * np.sqrt(S(y, y)))
    return result


def r2(x, y):
    return r(x, y) ** 2


def line(x, y, significance=10):
    b1 = S(x, y) / S(x, x)
    b0 = mean(y) - b1 * mean(x)
    return "y = " + str(round(b0, significance)) + " + " + str(round(b1, significance)) + "x"


def pred(x, y, val):
    b1 = S(x, y) / S(x, x)
    b0 = mean(y) - b1 * mean(x)
    return b0 + b1 * val


def SSE(x, y):
    n = len(x)
    result = 0
    for i in range(n):
        result += (y[i] - pred(x, y, x[i])) ** 2
    return result


def SST(x, y):
    n = len(x)
    result = 0
    for i in range(n):
        result += (y[i] - mean(y)) ** 2
    return result


def var(x, y):
    return SSE(x, y) / (len(x) - 2)


def std(x, y):
    return (var(x, y)) ** (0.5)

def plot(x, y, significance=4):
    x1 = np.linspace(np.min(x), np.max(x), 1000)
    significance = 4
    Z = loss(x, y)
    plotLoss(Z)
    plt.scatter(x, y, label="data")
    plt.title("Linear Regression: r^2=" + str(round(r2(x, y), significance)) + ", r=" + str(round(r(x, y), significance)))
    plt.plot(x1, pred(x, y, x1), '-r', label=line(x, y,significance))
    plt.text(np.median(x[int(len(x)/2):len(x)]), np.median(y[:int(len(y)/2)]), 'SSE=' + str(round((SSE(x, y)), significance)) + "\nSST=" + str(round(SST(x, y), significance)), horizontalalignment='center',  verticalalignment='center', fontsize=12)
    plt.legend(loc="upper left")
    plt.show()
    gradient_descent(x,y,Z)
#^^Largely From Simple Lin Reg (Jacob's) class, with changes here and there for functionality with my code.
#----------------------------------------------------------------------------------------------------------
def SSD(x_data, y_data):
#This finds the sum of squares distance
    index = 0
    #LM = 52 + 2x
    startSlope = 2
    startIntercept = 52
    z = startSlope * x_data + startIntercept
    print(z)
    d = y_data - z
    print("This is residuals: ", d)
    e = d * d
    sum = np.sum(e)
    print(sum)
    #for i in y_data:

def plotLine(slope,intercept):
    plt.scatter(x_data,y_data)
    x = np.linspace(0,60,100)
    y = slope*x+intercept
    label = "y = " + str(round(intercept, 4)) + " + " + str(round(slope, 4)) + "x"
    plt.plot(x, y, '-r', label="y=mx+b")
    plt.show()

def iterSlope(slope,intercept, x_data, y_data):
    for i in range(10):
        slope = slope + .2
        z = slope * x_data + intercept
        d = y_data - z
        e = d * d
        sum = np.sum(e)
        plotLine(slope,intercept)

def gradient_descent(X,y,theta, learning_rate=0,iterations=10):
       # X = 0 #Matrix of X with added bias unit
        #y = 0 #Vector of Y
        #theta = 0 #Vector of thetas np.random.random(10) Z
        learning_rate = 0
        iterations = 10 #number of iterations

        #Returns the final theta vector and array of cost history over n number of iterations
        n = len(y)
        cost_history = np.zeros(iterations)
        theta_history = np.zeros((iterations, 2))
        for it in range(iterations):
            prediction = np.dot(X,theta)
            theta = theta - (1/n)*learning_rate*(X.T.dot((prediction - y)))
            theta_history[it,ij] = theta.T
            cost_history[it] = cal_cost(theta,X,y)

        return theta, cost_history, theta_history

def loss(x_data, y_data):
    bb = np.arange(0, 100, 1)  # bias (intercept)
    print("This is bb:", bb)
    ww = np.arange(-5, 5, 0.1)  # weight (slope)
    print("This is ww:", ww)
    Z = np.zeros((len(bb), len(ww)))
    for i in range(len(bb)):
        for j in range(len(ww)):
            b = bb[i]
            w = ww[j]
            Z[j][i] = 0
            for n in range(len(x_data)):
                Z[j][i] = y_data[n] - (w * x_data[n] + b) #For Each combination of i/j value this represents the sum of distances to residuals for each line.
            Z[j][i] = (Z[j][i] * Z[j][i])/len(x_data) #makes all residuals positive
    print("Length of Z: ", len(Z))
    print("Length of x", len(x_data))
    print("Z[1][1]: ", Z[1][1])

    return Z



def plotLoss(Z):
    plt.plot(Z)
    plt.xlabel('Slope = Weight = b')
    plt.ylabel('Intercept = w')
    plt.title("Sum Distance Loss Function")  # Slope = x, Intercept = y
    plt.show()

def main():
    # Enter data
    x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
    x_dataDiff = np.array([10,20,30,40,50,60,70,88,99,100], float)
    y_data = 2 * x_data + 50 + 5 * np.random.random(10)  # updated

    plot(x_data, y_data)
    loss(x_data,y_data)

main()
#SSD(x_data,y_data)
#iterSlope(.5, 5, x_data,y_data) #Testing





