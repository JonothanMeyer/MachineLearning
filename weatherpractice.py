#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18  2019
@author: fengjiang
"""
from sklearn.metrics import classification_report
# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']

# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)

# converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

#combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(predicted)

#test

import pandas as pd
import numpy as np

# create a DataFrame
ODI_names = {'name': ['Tendulkar', 'Sangakkara', 'Ponting',
                     'Jayasurya', 'Jayawardene', 'Kohli',
                     'Haq', 'Kallis', 'Ganguly', 'Dravid']}

ODI_runs = {'runs': [18426, 14234, 13704, 13430, 12650,
                     11867, 11739, 11579, 11363, 10889]}
dfnames = pd.DataFrame(ODI_names)
dfruns = pd.DataFrame(ODI_runs)

# print the original DataFrame
print("Original DataFrame :")
print(dfnames)
print(dfruns)

# shuffle the DataFrame rows
dfnames = dfnames.sample(frac=1, random_state =5)
dfruns = dfruns.sample(frac=1, random_state= 5)

# print the shuffled DataFrame
print("\nShuffled DataFrame:")
print(dfnames)
print(dfruns)