# -*- coding: utf-8 -*-
"""
Created on June  16 2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# NumPy dizi (array) dönüşümü
X = x.values
Y = y.values


#linear regression / doğrusal model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
# görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
# 2. dereceden oluşturulmuş bir polinom
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
# görselleştirme
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# veriyi daha iyi gösterebilmek için dereceyi 2'den 4'e çıkardık
from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree = 4)
x_poly4 = poly_reg4.fit_transform(X)
print(x_poly4)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4,y)
# görselleştirme
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg4.predict(poly_reg4.fit_transform(X)), color = 'blue')
plt.show()


#tahminler
# linear regression ile tahmin
print("Predict with Linear Regression")
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

# polynomial regression ile tahmin
print("Predict with Polynomial Regression")
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

# 4. derece polynomial regression ile tahmin
print("Predict with 4th Degree Polynomial Regression")
print(lin_reg4.predict(poly_reg4.fit_transform([[11]])))
print(lin_reg4.predict(poly_reg4.fit_transform([[6.6]])))


# sonuçların dataların dağılımına göre değişmekle birlikte, polynomial regression ile daha doğru çıktığını görüyoruz









