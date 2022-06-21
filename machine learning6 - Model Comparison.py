# -*- coding: utf-8 -*-
"""
Created on June  17 2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score #yöntemleri karşılaştırmak için R2 değeri kullanacağız

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

# değerlendirme / karşılaştırma
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

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

# değerlendirme / karşılaştırma
print('Polynomial2 R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('Polynomial4 R2 degeri')
print(r2_score(Y, lin_reg4.predict(poly_reg4.fit_transform(X))))


# Support Vector Regression
# Bu regresyon çeşidi outlierlara karşı hassas olduğu için ölçeklendirme yapmak zorundayız
# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
#y_olcekli = sc2.fit_transform(Y)
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# Support Vector Regression
# Destek Vektör Regresyomu
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='blue')
plt.show()

# SVR ile tahmin
print("Predict with SVR")
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

# değerlendirme / karşılaştırma
print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

# Decision Tree
# Karar Ağacı ile Tahmin
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

# X +0.5 / X-0.4 için de aynı grafiği çiziyor çünkü bu değerler aynı aralığa düşüyor
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()

print("Predict with Decision Tree")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

# değerlendirme / karşılaştırma
print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))


# Random Forest
# Rassal Ağaçlar
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,Y.ravel())

print("Predict with Random Forest")
print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y, color='red')
plt.plot(X, rf_reg.predict(X), color='blue')
plt.plot(X, rf_reg.predict(Z), color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()
# Decision Tree bildiğimiz değerleri tahmin etmede başarılı
# fakat bilmediğimiz değerlerin tahmininde, bildiğimiz değerleri verme eğiliminde

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))


#Ozet R2 değerleri
print('-----------------------')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial2 R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('Polynomial4 R2 degeri')
print(r2_score(Y, lin_reg4.predict(poly_reg4.fit_transform(X))))

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))









