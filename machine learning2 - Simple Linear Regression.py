# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:16:14 2022

@author: Yunus
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

veriler = pd.read_csv("satislar.csv")
print (veriler)

aylar = veriler [["Aylar"]]
print(aylar)

satislar = veriler [["Satislar"]]
print(satislar)

#iloc = integer location
satislar2 = veriler.iloc[:,:1].values
print(satislar)

# data frame bölme
# x bağımsız değişken, y bağımlı değişken
from sklearn.model_selection import train_test_split

# genelde datanın %33 ü test için, %67'si train için bölünür
# random_state rastlantısal dağılım için
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size=0.33, random_state=0)

"""
# ölçekleme / scaling
# bu işlemin amacı, aralarında fark olan sayı gruplarını birbirine yakın değerlerde buluşturmak
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

#REGRESYON
#Basit Doğrusal Regresyon / Simple Linear Regression
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

# Görselleştirme
# verileri sıralama
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

















