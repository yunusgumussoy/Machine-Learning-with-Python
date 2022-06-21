# -*- coding: utf-8 -*-
"""
Created on June  19  2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('Iris.xls')
#pd.read_csv("veriler.csv")
#test
print(veriler)


x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
# print(y)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

# fit eğitim
X_train = sc.fit_transform(x_train)
# transform uygulama
X_test = sc.transform(x_test)

# SINIFLANDIRMA ALGORİTMALARI
# Logistic Regression 
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
# print(y_pred)
# print(y_test)

# Confusion Matrix
# Karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("LR")
print(cm)

# KNN Algoritması
# K Nearest Neighborhood
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric = 'minkowski') 
# komşu sayısını değiştirerek deneyebiliriz, bu örnekte az komşu ile daha başarılı tahminler oldu
# knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print ('KNN')
print (cm)

# Support Vector Machines
from sklearn.svm import SVC
svc = SVC(kernel='poly') # linear, rbf, sigmoid
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy') # Gini
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


# ROC - Receiver Operating Characteristic
# sonuç olasılıkları için
y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

# True Positive Rate, False Positive Rate değerleri 
from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)












