import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#1) Generate 500 random X values from -3 to 3
data_points = 500
X = np.random.uniform(-3,3,data_points)
X = X.reshape((len(X), 1))
#print("new X: ", X)
#print(type(X))
#print("X: ", X)

def checkInRange(x):
    for i in range(data_points):
        if x[i] < -3 or x[i] > 3:
            print("False")
            return False
    print(True)
    return True

#2) Generate 500 Y values using distribution "y = 0.5 * X^5 - X^3 - X^2 + 2 + a little bit randomness +/-
y = .5 * X**5 - X**3 - X**2 + 2 #+ np.random.normal(0,1,1)
for i in range(len(y)):
    randomnum = np.random.normal(0,5,1)
    y[i] = y[i] + 1 * randomnum




#3) Use X and Y as the whole dataset and use 200 samples as testing + 300 samples as training. Testing and training sets must be disjoint.
testing_X = X[:200]
print("length of testing_X: ", len(testing_X))
testing_y = y[:200]
print("Length of testing_y: ", len(testing_y))

training_X = X[200:500]
print("length of training_X: ", len(training_X))
training_y = y[200:500]
print("length of training_y:", len(training_y))


#4) Try Linear Regression and Polynomial Regression (PolynomialFeatures + LinearRegression) in SKLearn from degree 2 to 25 to fit the training data samples.
regr = LinearRegression()
regr.fit(training_X, training_y)
y_pred = regr.predict(testing_X)

model2 = LinearRegression()
poly2_features = PolynomialFeatures(degree=5)
X_poly2 = poly2_features.fit_transform(training_X)
X_poly2_test = poly2_features.fit_transform(testing_X)
model2.fit(X_poly2, training_y)
y_pred2 = model2.predict(X_poly2_test)



# The coefficients
#print('Coefficients: \n', regr.coef_)
#print('Itercept: \n', regr.intercept_)
# The mean squared error
#print('Mean squared error of linear model: %.2f'
 #     % mean_squared_error(testing_y, y_pred))
#print('Mean squared error of poly2 model: %.2f'
 #     % mean_squared_error(testing_y, y_pred2))
# The coefficient of determination: 1 is perfect prediction
#print('Coefficient of determination: %.2f'
  #    % r2_score(testing_y, y_pred))
#print('Coefficient of determination ploy2: %.2f'
 #     % r2_score(testing_y, y_pred2))

def MSE_array(x_data, y_data, nth_degree):
    iteration_array_history = []
    MSE_value_history = []
    for i in range(nth_degree):
        model2 = LinearRegression()
        poly2_features = PolynomialFeatures(degree=i)
        X_poly2 = poly2_features.fit_transform(x_data)
        model2.fit(X_poly2, y_data)
        y_pred2 = model2.predict(X_poly2)
        iteration_array_history.append(i)
        MSE = mean_squared_error(y_data, y_pred2)
        MSE_value_history.append(MSE)
    return iteration_array_history, MSE_value_history



    #print('Mean squared error of linear model: %.2f'
     #     % mean_squared_error(y_data, y_pred))
    #print('Coefficient of determination: %.2f'
     #     % r2_score(y_data, y_pred))
int = 4
print(type(int))

def plot_nth_degree(x_data, y_data, nth_degree,MSE_training_history, MSE_testing_history):
    if nth_degree > 0:
        model = LinearRegression()
        model.fit(x_data, y_data)
        y_pred = model.predict(x_data)
        model2 = LinearRegression()
        poly2_features = PolynomialFeatures(degree=nth_degree)
        X_poly2 = poly2_features.fit_transform(training_X)
        X_poly2_test = poly2_features.fit_transform(testing_X)
        model2.fit(X_poly2, training_y)

        poly_features = PolynomialFeatures(degree=nth_degree, include_bias=False)
        X_poly = poly_features.fit_transform(training_X)
        model = LinearRegression()
        model.fit(X_poly, training_y)
        Xplot = np.arange(-3, 3, .02)
        Xplot = Xplot.reshape(-1, 1)
        Xplot_poly = poly_features.fit_transform(Xplot)
        yplot_pred = model.predict(Xplot_poly)

        plt.scatter(x_data, y_data, color='red', label='Data Point', linewidths=1)
        label = "Poly Degree:" + str(nth_degree)
        plt.plot(x_data, y_pred, color='blue', linewidth=3, label='Poly Degree: 1')
        TrainL = MSE_history_training[-1]
        TestL =  MSE_history_testing[-1]
        plt.title("Machine Learnining, Quiz 2, Jonothan Meyer, TrainL:" + str(round(TrainL, 1)) + " TestL:" + str(round(TestL,1)))
        plt.plot(Xplot, yplot_pred, color='green', linewidth=3, label=label)
        plt.legend(loc="upper left")
        plt.show()

nth_degree = 20
iterationValues_training, MSE_history_training = MSE_array(training_X,training_y,nth_degree)
iterationValues_testing, MSE_history_testing = MSE_array(testing_X,testing_y,nth_degree)

plot_nth_degree(testing_X,testing_y,nth_degree,MSE_history_training,MSE_history_testing)



plt.plot(iterationValues_training, MSE_history_training, color='green',label='training')
plt.plot(iterationValues_testing, MSE_history_testing, color='red',label='testing')
plt.title('Machine Learnining, Quiz 2, Jonothan Meyer')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.show()