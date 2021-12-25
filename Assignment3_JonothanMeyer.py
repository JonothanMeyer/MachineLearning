# Jonothan Meyer
# Machine Learning
# HW 3
# 03/24/21
import cv2
import numpy as np
from imutils import paths
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import time

def load(imagePath_list, verbose=-1):
    data = []
    labels = []
    # loop over the input images
    for (i, imagePath) in enumerate(imagePath_list):
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        # treat our processed image as a "feature vector"
        # by updating the data list followed by the labels
        data.append(image)
        labels.append(label)
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
                                                  len(imagePath_list)))
    return np.array(data), np.array(labels)


def split_threeway(data, labels, random_state=5):
    """Used to partition data/labels into 3 sets: training (70%), testing (20%), and evaluation (10%)
    This is done by parsing data/labels into a panda data frame, then shuffling the rows of data/labels in an
    exact state using the 'random_state' argument. The proper proportion of data is taken for each respective
     variable (training/testing/evaluate X (data) and y (labels). The data is then parsed back into a numpy
     array and shaped properly before returned."""
    labels = pd.DataFrame(labels)
    data = pd.DataFrame(data)
    labels = labels.sample(frac=1, random_state=random_state)
    data = data.sample(frac=1, random_state=random_state)
    testing_X = data[:2100]
    testing_X = testing_X.to_numpy()
    testing_y = labels[:2100]
    testing_y = testing_y.to_numpy()
    testing_y = np.ravel(testing_y)
    training_X = data[2100:2700]
    training_X = training_X.to_numpy()
    training_y = labels[2100:2700]
    training_y = training_y.to_numpy()
    training_y = np.ravel(training_y)
    valid_X = data[2700:3000]
    valid_X = valid_X.to_numpy()
    valid_y = labels[2700:3000]
    valid_y = valid_y.to_numpy()
    valid_y = np.ravel(valid_y)
    return testing_X, training_X, testing_y, training_y, valid_X, valid_y


def checktypes(testing_X, training_X, testing_y, training_y, valid_X, valid_y):
    """My method of partitioning the data into 3 parts gave me some trouble since I was changing the data type of
    'data' and 'labels' from numpy.array > panda.dataframe > numpy.array > reshape(numpy.array). This function was used
    to check the data types of the partitions to make sure they would work properly for later functions"""
    print("testing_X: ", type(testing_X), testing_X.shape)
    print("training_ X: ", type(training_X), training_X.shape)
    print("testing_y: ", type(testing_y), testing_y.shape)
    print("training_y: ", type(training_y), training_y.shape)
    print("valid_X: ", type(valid_X), valid_X.shape)
    print("valid_y: ", type(valid_y), valid_y.shape)


def makeOdd(k):
    """With k-neighbor classification an odd number of k works best in instances of a tie. This function takes a k,
    rounds it to an int, checks to see if its odd, and if not increments it by 1 to make it odd. """
    k = round(k)
    if (k % 2) != 0:
        #print("k is already odd")
        return k
    else:
        k = k + 1
        #print("'k = k + 1' to make k odd")
        return k


def plotK(k_floor, k_ceiling, testing_X, training_X,testing_y, training_y):
    """This process of collecting accuracy scores and storing them in an array is from previous HW for this classs
     as well as information from 'towardsdatascience.com' and it's KNN page example. This finds the accuracey score
     from (k_floor, k_ceiling) and plots the progression"""
    start_time = time.perf_counter()
    k_range = range(k_floor, k_ceiling)
    scores = {}
    all_kscores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training_X, training_y)
        y_pred = knn.predict(testing_X)
        scores = metrics.accuracy_score(testing_y, y_pred)
        all_kscores.append(scores)
        if k == round(k_ceiling/4) or k == round(k_ceiling/2) or k == round(k_ceiling - k_ceiling/4):
            print("Calculated/Plotted: ", k, "/", k_ceiling)
    plt.plot(k_range, all_kscores)
    plt.xlabel("K")
    plt.ylabel("Testing Accuracy")
    plt.show()
    end_time = time.perf_counter()
    total_time = round((end_time - start_time)/60, 2)
    print("Time taken to Calculate/Plot: ", total_time, "min")
    max_value = max(all_kscores) #from 1-10, best is 3. from 1-50 best is 48. from 1-500, best is 489
    index = all_kscores.index(max_value)
    print("Best in range is K =", index-1)
    print("Best Accuracy: ", round(max_value, 4))


def predict(training_X, training_y, testing_X, k):
    num_test = testing_X.shape[0]
    # lets make sure that the output type matches the input type
    y_pred = np.zeros(num_test, dtype=training_y.dtype)
    for i in range(num_test):
        # L1_distances = np.sum(np.abs(training_X - testing_X[i, :]), axis=-1)
        L2_distances = np.sqrt(np.sum(np.square(training_X - testing_X[i,:]), axis = 1))
        # min_index = np.argmin(L1_distances)  # get the index with smallest distance
        K_index = np.argsort(L2_distances)[:k]
        # K_labels=np.zeros(len(K_index))
        K_labels = []
        for j in range(len(K_index)):
            K_labels.append(training_y[K_index[j]])
        y_pred[i] = max(set(K_labels), key=K_labels.count)
    return y_pred


def plotk_wpredict(k_floor, k_ceiling, testing_X, training_X,testing_y, training_y):
    """This runs the same cycle as the 'plotk' function, but uses the predict argument
    from Professor Jiang's Demo. Shows the time it takes to complete."""
    k_range = range(k_floor, k_ceiling)
    print("Plotting Accuracy of 'k' from", k_range)
    start_time = time.perf_counter()
    all_kscores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training_X, training_y)
        y_pred = predict(training_X, training_y, testing_X, k)
        scores = metrics.accuracy_score(testing_y, y_pred)
        all_kscores.append(scores)
        if k == round(k_ceiling/4) or k == round(k_ceiling/2) or k == round(k_ceiling - k_ceiling/4):
            print("Calculated/Plotted: ", k, "/", k_ceiling)
    end_time = time.perf_counter()
    plt.plot(k_range, all_kscores)
    plt.xlabel("K")
    plt.ylabel("Testing Accuracy")
    plt.show()
    total_time = round((end_time - start_time)/60, 2)
    print("Time taken to Calculate/Plot: ", total_time, "min")
    max_value = max(all_kscores)
    index = all_kscores.index(max_value)
    print("Best in range is K =", index - 1)
    print("Best Accuracy: ", round(max_value, 3))


def threefiveseven(testing_X, training_X, testing_y, training_y):
    for k in [3, 5, 7]:
        print("k = ", k)
        k = makeOdd(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training_X, training_y)
        y_pred = predict(training_X, training_y, testing_X, k)
        scores = classification_report(testing_y, y_pred)
        print(scores)
        print(round(metrics.accuracy_score(testing_y, y_pred), 3))


def main():
    # Get images and resize
    print("[INFO] loading images...")
    path = "C:\\Users\\Jonothan\\Desktop\\MSU-Spring2021\\Machine Learning\\Assignments\\HW3\\KNN\\KNN\\animals"
    imagePath_list = list(paths.list_images(path))
    (data, labels) = load(imagePath_list, verbose=500)
    data = data.reshape((data.shape[0], 3072))
    print("[INFO] features matrix: {:.1f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    # Split the data into 70% testing, 20% training, and 10% validation
    testing_X, training_X, testing_y, training_y, valid_X, valid_y = split_threeway(data, labels, 5)
    # checktypes(testing_X, training_X, testing_y, training_y, valid_X, valid_y)

    # Converter 'labels' string values to numerical
    le = preprocessing.LabelEncoder()
    training_y = le.fit_transform(training_y)
    testing_y = le.fit_transform(testing_y)

    # Train the Classifier and evaluate for k = 3, 5, 7
    threefiveseven(testing_X, training_X, testing_y, training_y)
    # Best distance to use is L2 w best accuracy at k=7
    # Best accuracy for L1 is k=3


    # Plot the accuracy of k from 1-10)
    # plotK(1, 50, testing_X, training_X, testing_y, training_y) #This plots (k_floor, k_ceiling) with the built in
    # predict() function from sklearn
    plotk_wpredict(1, 10, testing_X, training_X, testing_y, training_y) #L1 best k: 3, L2 best k: 7,


main()
