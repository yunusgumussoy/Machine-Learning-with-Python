# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 01:44:45 2022

@author: Yunus
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
dataset = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(dataset)

#veri on isleme
# modelin datayı ezberlemesini engellemek için RowNumber, CustomerId, ve Surname gibi sütunları çıkarıyoruz
X = dataset.iloc[:,3:13].values
# müşterinin terkedip etmeyeceğini Exited sütunu ile tahmin etmeye çalışacağız
Y = dataset.iloc[:,13].values


#encoder: Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Geographic ve Gender değişkenlerini numerik değerlere dönüştürüyoruz
labelencoder_X_1 = LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()

X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Birden fazla sütunun aynı anda ayrı ayrı dönüştürülmesi için One Hot Encoding kullanıyoruz
# BÖylece değişkenleri sayısal değerlere dönüştürüyoruz
ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)
















