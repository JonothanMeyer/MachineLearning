# import the necessary packages
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np


def predict(training_X, training_y, testing_X, k):
    num_test = testing_X.shape[0]
    # lets make sure that the output type matches the input type
    y_pred = np.zeros(num_test, dtype=training_y.dtype)
    for i in range(num_test):
        L1_distances = np.sum(np.abs(training_X - testing_X[i, :]), axis=-1)
        # L2_distances = np.sqrt(np.sum(np.square(trainX - testX[i,:]), axis = 1))
        # min_index = np.argmin(L1_distances)  # get the index with smallest distance
        K_index = np.argsort(L1_distances)[:k]
        # K_labels=np.zeros(len(K_index))
        K_labels = []
        for j in range(len(K_index)):
            K_labels.append(training_y[K_index[j]])
        y_pred[i] = max(set(K_labels), key=K_labels.count)
    return y_pred
