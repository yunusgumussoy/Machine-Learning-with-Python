#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 2022

@author: Yunus GÜMÜŞSOY
"""

#1. Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme
#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
# print (veriler)

#veri on isleme
#encoder:  Kategorik -> Numeric
# tek tek değişkenleri encode etmek yerine, tüm değişkenleri encode ediyoruz
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)
# print (veriler2)
# buradan da görüleceği üzere halihazırda numerik olan değişkenler de encode edildi

# bunu düzeltmek için öncelikle outlook değişkenini overcast, rainy, ve sunny olarak dağıtacağız
c = veriler2.iloc[:,:1]
#print (c)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

# c olarak tanımladığımız outlook değişkenini overcast, rainy, ve sunny sütunlarıyla data frame yaptık
havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])

# temperature ve humidity zaten numerik olduğu için veriler'den onları direkt alıyoruz ve havadurumu ile birleştiriyoruz
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)

# veriler2'de windy ve play değişkenleri zaten doğru encode edilmişti, son 2 sütunu direkt alıyoruz ve son veriler ile birleştiriyoruz
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)
print (sonveriler)



# verilerin egitim ve test icin bolunmesi
# humidity tahmini için
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)

print(y_pred)


# Model Doğrulama
# Geri Eleme Yöntemi / Backward Elimination
import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )

# Bütün değişkenleri modele ekliyoruz
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

# buradaki sonuca göre x1'in p değerinin (0.593) en yüksek olduğunu gördük, onu modelden çıkaracağız
sonveriler = sonveriler.iloc[:,1:]
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

# hala p değerlerinin çok büyük olduğunu görüyoruz
# x1 yani windy değişkenini data frame den çıkararak, tekrar tahmin yaptıracağız
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train, y_train)


y_pred2 = regressor.predict(x_test)







