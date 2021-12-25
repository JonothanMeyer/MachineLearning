import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#1) Generate 500 random X values from -3 to 3
data_points = 500
X = np.random.uniform(-3,3,data_points)
X = X.reshape((len(X), 1))
print("new X: ", X)
print(type(X))
print("X: ", X)

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
    randomnum = np.random.normal(0,10,1)
    y[i] = y[i] + randomnum




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
# Create linear regression object
model = LinearRegression()
# Train the model using the training sets
model.fit(training_X, training_y)
# Make predictions using the testing set
y_pred = model.predict(testing_X)

model2 = LinearRegression()
poly2_features = PolynomialFeatures(degree=1)
X_poly2 = poly2_features.fit_transform(training_X)
X_poly2_test = poly2_features.fit_transform(testing_X)
model2.fit(X_poly2, training_y)
y_pred2 = model2.predict(X_poly2_test)

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(training_X)
model = LinearRegression()
model.fit(X_poly, training_y)
Xplot = np.arange(-3.5, 3.5, .02)
Xplot = Xplot.reshape(-1, 1)
Xplot_poly = poly_features.fit_transform(Xplot)
yplot_pred = model.predict(Xplot_poly)

plt.scatter(testing_X,testing_y, color='red', linewidths=1)
plt.plot(testing_X, y_pred, color='blue', linewidth=3)
plt.plot(Xplot, yplot_pred, color='green', linewidth=3)
plt.show()

# The coefficients
#print('Coefficients: \n', regr.coef_)
#print('Itercept: \n', regr.intercept_)
# The mean squared error
print('Mean squared error of linear model: %.2f'
      % mean_squared_error(testing_y, y_pred))
print('Mean squared error of poly2 model: %.2f'
      % mean_squared_error(testing_y, y_pred2))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(testing_y, y_pred))
print('Coefficient of determination ploy2: %.2f'
      % r2_score(testing_y, y_pred2))

def MSE():
    print(things)













