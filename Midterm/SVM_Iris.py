import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#data=np.array(pd.read_csv('../Data/MNIST/train.csv'))
#print("[INFO] evaluating classifier...")
#trainX=data[0:100,1:]
#trainY=data[0:100,0]

data = np.array(
    pd.read_csv('C:\\Users\\Jonothan\\Desktop\\MSU-Spring2021\\Machine Learning\\Assignments\\Midterm\\iris.csv'))
print("[INFO] evaluating classifier...")
X=data[:,:-1]
y=data[:,-1]
le = LabelEncoder()
ylabels = le.fit_transform(y)
trainX,testX,trainY,testY=train_test_split(X,ylabels,test_size=0.4,random_state=0)

Gamma=0.001
C=1#0.001
model =  svm.SVC(kernel='linear', C=C,gamma=Gamma)
#model = LogisticRegression()
#model =  DecisionTreeClassifier()
model.fit(trainX, trainY)
predY=model.predict(testX)
print(classification_report(testY, predY))
#
#Showlist=np.arange(10)
#for i in Showlist:
#    sample=testX[i]
#    sample=sample.reshape((28,28))
#    plt.imshow(sample,cmap='gray')
#    plt.title('The prediction:'+str(predY[i]))
#    plt.show()