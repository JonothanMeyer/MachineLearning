import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# column_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
file = "C:\\Users\\Jonothan\\Desktop\\MSU-Spring2021\\Machine Learning\\Assignments\\Midterm\\iris.csv"
iris_data = np.array(pd.read_csv(file))  # , names=column_names))

X = iris_data[:, :-1]
y = iris_data[:, -1]


le = LabelEncoder()
ylabels = le.fit_transform(y)


C = 1
test_size = 0.8
random_state = 0 #randint(0,1000)

def accuracy_today(testY, predY):
    """Using the testingY and predictedY data this finds and returns the accuracy as XX.XXX%"""
    ac = accuracy_score(testY, predY, normalize=True)
    ac = ac * 100
    ac = round(ac, 3)
    return ac

def lin_svc(trainX, trainY, testX, testY):
    """Using the testing and training sets this constructs a linear model. predictors have already been
    choosen, and the training/testing data has already been assigned. Returns the accuracy of this model w/ previously
    choosen predictors"""
    Gamma = 0.001
    model = svm.SVC(kernel='linear', C=C, gamma=Gamma)
    model.fit(trainX, trainY)
    predY=model.predict(testX)
    #print("Linear SVC Classification Report:")
    #print(classification_report(testY, predY))
    cnf_matrix = metrics.confusion_matrix(testY, predY)
    accuracy = accuracy_today(testY, predY)
    return accuracy

def nonlin_svc(trainX, trainY, testX, testY):
    """Using the testing and training sets this constructs a non-linear model (poly = 3). predictors have already been
    choosen, and the training/testing data has already been assigned. Returns the accuracy of this model w/ previously
    choosen predictors"""
    model = svm.SVC(kernel='poly', degree=3, C=C)
    model = model.fit(trainX, trainY)
    predY = model.predict(testX)
    #print("NonLinear SVC Classification Report:")
    #print(classification_report(testY, predY))
    cnf_matrix = metrics.confusion_matrix(testY, predY)
    accuracy = accuracy_today(testY, predY)
    return accuracy


def logreg(trainX, trainY, testX, testY):
    """Using the testing and training sets this constructs a logistic regression model. predictors have already been
    choosen, and the training/testing data has already been assigned. Returns the accuracy of this model w/ previously
    choosen predictors"""
    # Logistic Regression
    #trainX = trainX[:,(2,3)]  # petal width
    #testX = testX[:,(2,3)]
    model = LogisticRegression(multi_class= "auto", solver="lbfgs", C=C)
    model.fit(trainX,trainY)
    predY = model.predict(testX)
    #print("Logistic Regression Classification Report:")
    #print(classification_report(testY, predY))
    cnf_matrix = metrics.confusion_matrix(testY, predY)
    accuracy = accuracy_today(testY, predY)
    return accuracy

def find_best_lin_svc():
    """Constructs a linear model for all combinations of predictors and returns the predictor
    combo sequence, and the associated accuracy using those predictors
    predictor_history = array of different predictors used (pred_one, pred_two)
    accuracy_history = array of sequence of how accurate each predictor model was"""
    predictor_history = []
    accuracy_history = []
    for i in range(4):
        for j in range(4):
            if i >= j:# Used to avoid modeling the same predictors twice. ex. (3,2) & (2,3) <- Redundant
                continue
            trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=test_size, random_state=random_state)
            pred_var_one = i
            pred_var_two = j
            trainX = trainX[:, (pred_var_one, pred_var_two)]
            testX = testX[:, (pred_var_one, pred_var_two)]
            accuracy = lin_svc(trainX, trainY, testX, testY)
            accuracy_history.append(accuracy)
            predictor_history.append((i, j))
    return predictor_history, accuracy_history

def find_best_nonlin_svc():
    """Constructs a non-linear (poly = 3) model for all combinations of predictors and returns the predictor
    combo sequence, and the associated accuracy using those predictors
    predictor_history = array of different predictors used (pred_one, pred_two)
    accuracy_history = array of sequence of how accurate each predictor model was"""
    predictor_history = []
    accuracy_history = []
    for i in range(4):
        for j in range(4):
            if i >= j:# Used to avoid modeling the same predictors twice. ex. (3,2) & (2,3) <- Redundant
                continue
            trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=test_size, random_state=random_state)
            pred_var_one = i
            pred_var_two = j
            trainX = trainX[:, (pred_var_one, pred_var_two)]  # petal width
            testX = testX[:, (pred_var_one, pred_var_two)]
            accuracy = nonlin_svc(trainX, trainY, testX, testY)
            accuracy_history.append(accuracy)
            predictor_history.append((i, j))
    return predictor_history, accuracy_history

def find_best_logreg():
    """Constructs a Logistic Regression model for all combinations of predictors and returns the predictor
    combo sequence, and the associated accuracy using those predictors
    predictor_history = array of different predictors used (pred_one, pred_two)
    accuracy_history = array of sequence of how accurate each predictor model was"""
    predictor_history = []
    accuracy_history = []
    for i in range(4):
        for j in range(4):
            if i >= j: # Used to avoid modeling the same predictors twice. ex. (3,2) & (2,3) <- Redundant
                continue
            trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=test_size, random_state=random_state)
            pred_var_one = i
            pred_var_two = j
            trainX = trainX[:, (pred_var_one, pred_var_two)]
            testX = testX[:, (pred_var_one, pred_var_two)]
            accuracy = logreg(trainX, trainY, testX, testY)
            accuracy_history.append(accuracy)
            predictor_history.append((i, j))
    return predictor_history, accuracy_history

def find_best_predictor(ls_accuracy_history, nls_accuracy_history, lr_accuracy_history):
    """This method searches through the accuracy history of all combinations of methods and predictors, and from
    that evaluates which method and predictor combo gives the highest accuracy
    best_method = The type of model that gives the highest accuracy
    max_value = the index number in the history of predictors that has the highest accuracy"""
    max_value_ar = []
    max_value1 = max(ls_accuracy_history)
    max_value_ar.append(max_value1)
    max_value2 = max(nls_accuracy_history)
    max_value_ar.append(max_value2)
    max_value3 = max(lr_accuracy_history)
    max_value_ar.append(max_value3)
    best_method = max_value_ar.index(max(max_value_ar))
    if(best_method == 0):
        max_value = ls_accuracy_history.index(max_value1)
        return best_method, max_value
    elif(best_method == 1):
        max_value = nls_accuracy_history.index(max_value2)
        return best_method, max_value
    elif(best_method == 2):
        max_value = lr_accuracy_history.index(max_value3)
        return best_method, max_value
    else:
        print("Error in Best method")
        return

def print_best_method(best_method, best_method_index, ls_predictor_history, nls_predictor_history, lr_predictor_history,
                      ls_accuracy_history, nls_accuracy_history,lr_accuracy_history):
    """This method assumes that the best method and predictors have already been found and converts/prints that
    information in a way that explicitly shows which method is best for the data set, the predictors used, and
    how accurate that method is."""
    print("-----------------------------------------------------------------------------------------------------------")
    print("Best Method and Predictors Found: ")
    if best_method == 0:
        print("Linear SVC")
        print("Hyperparameters:  ", ls_predictor_history[best_method_index])
        print('Accuracy: {:>15}%'.format(ls_accuracy_history[best_method_index]))
    elif best_method == 1:
        print("Non-Linear SVC")
        print("Hyperparameters:  ", nls_predictor_history[best_method_index])
        print('Accuracy: {:>15}%'.format(nls_accuracy_history[best_method_index]))
    elif best_method == 2:
        print("Logistic Regression")
        print("Hyperparameters:  ", lr_predictor_history[best_method_index])
        print('Accuracy: {:>15}%'.format(lr_accuracy_history[best_method_index]))


def print_class_report(best_method, best_method_index, ls_predictor_history, nls_predictor_history,
                       lr_predictor_history):
    """This method assumes that the best model and predictors are already found and takes them the first two argument.
    Using that information it rebuilds that particular model using those parameters and prints out the classification
    report."""
    print("-------------------------------Using Best Model and Predictors---------------------------------------------")
    if best_method == 0:
        temp = ls_predictor_history[best_method_index]
        pred_var_one = temp[0]
        pred_var_two = temp[1]
        trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=test_size, random_state=random_state)
        trainX = trainX[:, (pred_var_one, pred_var_two)]
        testX = testX[:, (pred_var_one, pred_var_two)]
        print("Linear SVC Classifiication Report")
        print("Hyperparameters: ", nls_predictor_history[best_method_index])
        Gamma = 0.001
        model = svm.SVC(kernel='linear', C=C, gamma=Gamma)
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        print(classification_report(testY, predY))
        return testY, predY
    elif best_method == 1:
        temp = ls_predictor_history[best_method_index]
        pred_var_one = temp[0]
        pred_var_two = temp[1]
        trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=test_size, random_state=random_state)
        trainX = trainX[:, (pred_var_one, pred_var_two)]
        testX = testX[:, (pred_var_one, pred_var_two)]
        print("Non-Linear SVC Classification Report")
        print("Hyperparameters: ", nls_predictor_history[best_method_index])
        model = svm.SVC(kernel='poly', degree=3, C=C)
        model = model.fit(trainX, trainY)
        predY = model.predict(testX)
        print(classification_report(testY, predY))
        return testY, predY
    elif best_method == 2:
        temp = lr_predictor_history[best_method_index]
        pred_var_one = temp[0]
        pred_var_two = temp[1]
        trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=test_size, random_state=random_state)
        trainX = trainX[:, (pred_var_one, pred_var_two)]
        testX = testX[:, (pred_var_one, pred_var_two)]
        print("Logistic Regression Classification Report")
        print("Hyperparameters: ", lr_predictor_history[best_method_index])
        model = LogisticRegression(multi_class="auto", solver="lbfgs", C=1.0)
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        print(classification_report(testY, predY))
        return testY, predY

def number_six(y_test, y_pred):
    """Borrowed heavily from the '2_Logostic_ExSKLearn_Demo.py' class and repurposed to be used with the 'pima' data"""
    print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(metrics.precision_score(y_test, y_pred, average='macro'),3))
    print("Recall:", round(metrics.recall_score(y_test, y_pred, average='macro'),3))
    print("F-measure:", round(metrics.f1_score(y_test, y_pred, average='macro'),3))

def plot_ConfMatrix(cnf_matrix):
    """Shows an easily readable Confusion Matrix for the model. Largely utilized and repurposed code from assignment 2
    and the '2_logistic_ExSKLearn_Demo.py' demo code. This presents the accuracy of the machine model through
     visual color shading.
    """
    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
    #ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    ls_predictor_history, ls_accuracy_history = find_best_lin_svc()
    print("Predictor History: ", ls_predictor_history)
    print("LS accuracy history: ", ls_accuracy_history)
    nls_predictor_history, nls_accuracy_history = find_best_nonlin_svc()
    print("NLS accuracy history: ", nls_accuracy_history)
    lr_predictor_history, lr_accuracy_history = find_best_logreg()
    print("LR accuracy history: ", lr_accuracy_history)
    all_accuracys = np.concatenate([ls_accuracy_history, nls_accuracy_history, lr_accuracy_history])
    print("Total Mean Accuracy:", round(all_accuracys.mean(axis=0), 2),"%")
    best_method, best_method_index = find_best_predictor(ls_accuracy_history, nls_accuracy_history, lr_accuracy_history)
    print_best_method(best_method, best_method_index,ls_predictor_history,nls_predictor_history,lr_predictor_history,
                      ls_accuracy_history, nls_accuracy_history, lr_accuracy_history)
    testY, predY = print_class_report(best_method, best_method_index, ls_predictor_history, nls_predictor_history,
                       lr_predictor_history)
    number_six(testY,predY)
    cnf_matrix = metrics.confusion_matrix(testY, predY)
    plot_ConfMatrix(cnf_matrix)


main()