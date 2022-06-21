# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:16:14 2022

@author: Yunus
"""
#kütüphaneler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#Çoklu Doğrusal Regresyon
#Multiple Linear Regression


veriler = pd.read_csv("veriler.csv")
print (veriler)

#eksik veri olmadığı için, eksik veri tamamlama kısmını atlıyoruz

# ülke ve cinsiyet değişkenleri kategorik olduğu için onları numerik değerlere çevireceğiz

# KATEGORİK DEĞİŞKENLERİ DÖNÜŞTÜRME

# 1. ülke değişkeni için / kategorik değişkenleri dönüştürme
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

# 2. cinsiyetleri kategorikten numerik değerlere dönüştüreceğiz
c = veriler.iloc[:,-1:].values
print (c)

from sklearn import preprocessing
# label encoding
le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print (c)

# one hot encoder
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print (c)

# 3. boy / kilo / yas değişkenleri zaten numerik, onları direkt ekleyeceğiz
Yas = veriler.iloc[:,1:4].values
print(Yas)

# NUMPY DİZİLERİNİ DATA FRAME E DÖNÜŞTÜRME
# data frame / ülke
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ["fr", "tr", "us"])
print (sonuc)

# data frame / boy, kilo, yaş
sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ["boy", "kilo", "yas"])
print (sonuc2)

# data frame cinsiyet (kategorikte numeriğe dönüşmüş hali c)
sonuc3 = pd.DataFrame(data=c[:,:1], index = range(22), columns = ["cinsiyet"])
print (sonuc3)

# data frame lerin birleştirilmesi
# concat refers to concatenate
# axis = 1 ile 1.satırdan itibaren data frameleri yan yana birleştiriyoruz (sütun olarak ekler)
s = pd.concat([sonuc,sonuc2], axis=1)
print (s)

s2 = pd.concat([s,sonuc3], axis=1)
print (s2)

# REGRESYON
# data frame bölme
# x bağımsız değişken, y bağımlı değişken
from sklearn.model_selection import train_test_split

# genelde datanın %33 ü test için, %67'si train için bölünür
# random_state rastlantısal dağılım için
# sonuc3'te cinsiyet var, cinsiyeti tahmin etmek istediğimiz için sonuc3 yazıyoruz
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3, test_size=0.33, random_state=0)

# REGRESYON MODELİ

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) #x bağımsız değişkenler ile y bağımlı değişkeni eğitiyoruz

y_pred = regressor.predict(x_test) #x bağımsız değişkenlere göre y bağımlı değişkeni tahmin ediyoruz

# boy tahmini için boy değişkeninin bulunduğu sütunu s2 data frame inden çekiyoruz
boy = s2.iloc[:,3:4].values
print(boy)

# s2 data frame inden boy değişkeninin dışında kalan kısmı, eğitim verisi için tekrar birleştiriyoruz
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol, sag], axis=1)

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)
y_pred2 = regressor2.predict(x_test)



# MODEL VE DEĞİŞKENLERİN BAŞARISINI ÖLÇMEK İÇİN
# Geri Eleme Yöntemi
#kütüphane
import statsmodels.api as sm

# regreson formülünde y=β0+β1X+ϵ, ϵ'yi bir sütun olarak ekliyoruz
# içinde 1 olan 22 satır 1 sütundan oluşan data frame i veri'ye ekliyoruz
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

# öncelikle bütün değişkenleri modele dahil ediyoruz
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# buradaki sonuca göre x5'in p değerinin (0.717) en yüksek olduğunu gördük, onu modelden çıkaracağız
X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# buradaki sonuca göre x5'in p değerinin (0.031) en yüksek olduğunu gördük, onu modelden çıkaracağız
X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())




















