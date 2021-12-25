#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fengjiang
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#from sklearn.cross_validation import train_test_split


X_train= np.array([[1,2],[1,2],[1,2],[2,3],[2,3]])
print(X_train.shape)
y_train=np.array([1,1,1,0,0])
X_test= np.array([[1,2],[1,2],[2,3],[1,2],[2,3]])
y_test=np.array([1,0, 0, 1, 0])
print("y test shape", y_test.shape)



logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)
#
y_pred=logreg.predict(X_test)
#print(y_pred)
print("X_test: ", X_test)
print("y_test: ", y_test)
print("accuracy score using sklearn model score",logreg.score(X_test,y_test))
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
#ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print(metrics.classification_report(y_test,y_pred))
plt.figure()
y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()