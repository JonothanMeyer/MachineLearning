#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USAGE
# python knn.py --dataset ../datasets/animals

# import the necessary packages
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import cv2
import os
import pandas as pd


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

        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
                                                  len(imagePath_list)))

    # return a tuple of the data and labels
    return (np.array(data), np.array(labels))


# grab the list of images that we'll be describing
print("[INFO] loading images...")
path = "C:\\Users\\Jonothan\\Desktop\\MSU-Spring2021\\Machine Learning\\Assignments\\HW3\\KNN\\KNN\\animals"
imagePath_list = list(paths.list_images(path))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
(data, labels) = load(imagePath_list, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))

print(data[2000])
print(labels[2000])
# print(len(labels))
# print(len(data[4]))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, train_size=0.7, random_state=0)
print(len(X_train), len(X_test))
print(X_train.shape)
print(y_test.shape)


def split_threeway(data, labels, testing_percent=0.7, training_percent=0.2, valid_percent=0.1):
	labels = pd.DataFrame(labels)
	data = pd.DataFrame(data)
	labels = labels.sample(frac=1, random_state=5)
	data = data.sample(frac=1, random_state=5)
	testing_X = data[:2100]
	print("length of testing_X: ", len(testing_X))
	testing_X = testing_X.reshape((len(testing_X), 1))
    testing_y = labels[:2100]
	print("Length of testing_y: ", len(testing_y))
    testing_y = testing_y.reshape((len(testing_y), 1))
    training_X = data[2100:2700]
    print("length of training_X: ", len(training_X))
    training_X = training_X.reshape((len(training_X), 1))
    training_y = labels[2100:2700]
    print("length of training_y:", len(training_y))
    valid_X = data[2700:3000]
    print("length of valid_X: ", len(valid_X))
    valid_y = labels[2700:3000]
    print("length of valid_y: ", len(valid_y))
    return testing_X, training_X, testing_y, training_y, valid_X, valid_y


def predict(training_X, training_y, testing_X, k):
    num_test = testing_X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype=training_y.dtype)
    for i in range(num_test):
        L1_distances = np.sum(np.abs(training_X - testing_X[i, :]), axis=-1)
        # L2_distances = np.sqrt(np.sum(np.square(trainX - testX[i,:]), axis = 1))
        # min_index = np.argmin(L1_distances)  # get the index with smallest distance
        K_index = np.argsort(L1_distances)[:k]
        # K_labels=np.zeros(len(K_index))
        K_labels = []
        for j in range(len(K_index)):
            K_labels.append(training_y[K_index[j]])
        Ypred[i] = max(set(K_labels), key=K_labels.count)
    return Ypred


training_X, testing_X, training_y, testing_y = train_test_split(data, labels, test_size=0.2, train_size=0.7,
                                                                random_state=0)
print(len(X_train), len(X_test))
print(X_train.shape)
print(y_test.shape)

k = round(math.sqrt(3000))
# testing_X, training_X, testing_y, training_y, valid_X, valid_y = split_threeway(data,labels)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(training_X, training_y)
y_pred = knn.predict(testing_X)

scores = classification_report(testing_y, y_pred)
print("scores: ", scores)

y_pred = predict(training_X, training_y, testing_X, k)
print("ypred: ", y_pred)

from sklearn import preprocessing, metrics

# creating labelEncoder
le = preprocessing.LabelEncoder()
animals_encoded = le.fit_transform(y_pred)
print(animals_encoded)
