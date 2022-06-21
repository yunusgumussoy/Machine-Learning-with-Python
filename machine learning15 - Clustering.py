# -*- coding: utf-8 -*-
"""
Created on June  20  2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('musteriler.csv')

# müşterilerin iş hacmi ve maaşını alıyoruz
X = veriler.iloc[:,3:].values


#3. K Means
from sklearn.cluster import KMeans

# 3 tane cluster noktası istiyoruz, algoritma olarak da k-means++ ı seçiyoruz
kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
# training
kmeans.fit_predict(X)


# cluster noktalarını yazdıyoruz
print(kmeans.cluster_centers_)

# burada bir döngü yardımıyla k-means algoritmasının datamız için optimal cluster sayısını görüyoruz
sonuclar = []
for i in range(1,25):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) #

plt.plot(range(1,25),sonuclar)
plt.show()

# 4 clusterlı K-Means sonuçları
kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c="red")
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c="blue")
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c="green")
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c="yellow")
plt.title("KMeans")
plt.show()


# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering (n_clusters = 4, affinity="euclidean", linkage="ward")
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c="red")
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c="blue")
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c="green")
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c="yellow")
plt.title("Hierarchical Clustering")
plt.show()

# dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()



