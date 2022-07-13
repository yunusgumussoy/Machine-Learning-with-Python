# -*- coding: utf-8 -*-
"""
Created on June  24 2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme
# modelin datayı ezberlemesini engellemek için RowNumber, CustomerId, ve Surname gibi sütunları çıkarıyoruz
X= veriler.iloc[:,3:13].values
# müşterinin terkedip etmeyeceğini Exited sütunu ile tahmin etmeye çalışacağız
Y = veriler.iloc[:,13].values


#encoder: Kategorik -> Numeric
from sklearn import preprocessing

# Geographic ve Gender değişkenlerini numerik değerlere dönüştürüyoruz
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])


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

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#3. Yapay Sinir ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

# katman / layer ekliyoruz
classifier = Sequential()

# giriş katmanımız 11 nerondan oluşuyor, gizli katmanında 6 nöron var
# güncel keras init = 'uniform' gerektirmiyor, eski keraslarda bu classifier a ekleniyor
# classifier.add(Dense(6, init = 'uniform', activation = 'relu' , input_dim = 11))

classifier.add(Dense(6, activation = 'relu' , input_dim = 11))
# ilk katmanda input dimension eklememiz gerekiyor, fakat sonraki katmanlar için gerek yok
classifier.add(Dense(6, activation = 'relu'))
"""
classifier.add(Dense(6, activation = 'relu'))
classifier.add(Dense(5, activation = 'relu'))
"""

# çıkış katmanında relu yerine sigmoid fonksiyonunu kullanıyoruz, çıkış outputunu da 1 olarak tanımlıyoruz
classifier.add(Dense(1, activation = 'sigmoid'))
# gizli katmanlardaki nöron sayısı aslında tahmine dayalı, giriş ve çıkış katmanlarının ortalaması olabilir


# neural networkumuzu compile ediyoruz
# tahmin etmeye çalıştığımız değişkenin verileri binary / 0-1 olduğu için binary_crossentropy kullanıyoruz
# kategorik olsaydı categorical_crossentropy i kullanacaktık
# metrics de neyi optimize edeceğimizi gösteriyor, accuracy i seçtik
classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

# epochs kaç sefer / aşamada öğreneceğini gösteriyor
classifier.fit(X_train, y_train, epochs=80)

# x_test e bakarak tahmin yap, bunu da y_pred e yaz
y_pred = classifier.predict(X_test)


y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)

