# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Machine Learning Course file by Yunus GÜMÜŞSOY.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # functions for creating and removing a directory (folder)
import seaborn as sns
import matplotlib.pyplot as plt
"""import warnings 
warnings.filterwarnings("ignore")"""

"""
veriler = pd.read_csv("veriler.csv")
print(veriler)

boy = veriler [["boy"]]
print(boy)

boykilo = veriler [["boy", "kilo"]]
print(boykilo)


#object oriented 
class insan:
    boy = 180
    def kosmak (self, b):
        return b + 10
    
ali = insan()
print (ali.boy)
print (ali.kosmak(90))

l = [1,3,4] #liste
"""
# 1. VERİ ÖN İŞLEME
# eksik veriler
# sci-kit learn

veriler = pd.read_csv("eksikveriler.csv")
print (veriler)

## yöntemlerden bir tanesi eksik veri yerine o sütunun / satırın ortalamasını eklemektir
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#iloc = integer location
Yas = veriler.iloc[:,1:4].values
print(Yas)
#fit fonksiyonu eğitmek için kullanılır, sayısal kolonları öğrenecek
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print (Yas)

#kategorik değişkenleri dönüştürme
ulke = veriler.iloc[:,0:1].values
print (ulke)

from sklearn import preprocessing
# label encoding
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print (ulke)

# one hot encoder
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print (ulke)

# 2. NUMPY DİZİLERİNİ DATA FRAME E DÖNÜŞTÜRME
# data frame / ülke
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ["fr", "tr", "us"])
print (sonuc)

# data frame / boy, kilo, yaş
sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ["boy", "kilo", "yas"])
print (sonuc2)

# data frame / cinsiyet
# -1 son sütun
cinsiyet = veriler.iloc[:,-1].values
print (cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ["cinsiyet"])
print (sonuc3)

# data frame lerin birleştirilmesi
# concat refers to concatenate
# axis = 1 ile 1.satırdan itibaren data frameleri yan yana birleştiriyoruz (sütun olarak ekler)
s = pd.concat([sonuc,sonuc2], axis=1)
print (s)

s2 = pd.concat([s,sonuc3], axis=1)
print (s2)


# data frame bölme
# x bağımsız değişken, y bağımlı değişken
from sklearn.model_selection import train_test_split

# genelde datanın %33 ü test için, %67'si train için bölünür
# random_state rastlantısal dağılım için
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3, test_size=0.33, random_state=0)


# ölçekleme / scaling
# bu işlemin amacı, aralarında fark olan sayı gruplarını birbirine yakın değerlerde buluşturmak
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

