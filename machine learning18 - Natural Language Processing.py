#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21 11:00:54 2022

@author: Yunus Gümüşsoy
"""

import numpy as np
import pandas as pd

col_names = ["Review", "Liked"]
yorumlar = pd.read_csv('Restaurant_Reviews.csv', names=col_names)


import re
import nltk

# kelimeleri kökenine ayırmak için
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# anlam ifade etmeyen kelimeleri temizlemek için
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing (Önişleme)
#Alfanumerik karakterlerin filtrelenmesi
derlem = []
for i in range(1000):
    # a-z veya A-Z içermeyen yorumları " " (boşluk) ile değiştir
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    # yorumları küçük harfe dönüştürme
    yorum = yorum.lower()
    # yorumlardaki kelimeleri bölerek, liste haline getirme
    yorum = yorum.split()
    # eğer yorumdaki kelime stopword değilse kökenine ayır, ve yeni halini listeye ekle
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    # işlenmiş kelimeleri tekrar birleştiriyoruz
    yorum = ' '.join(yorum)
    derlem.append(yorum)



#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken

# Machine Learning
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


















