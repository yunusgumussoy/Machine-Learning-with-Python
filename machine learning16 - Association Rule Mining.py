# -*- coding: utf-8 -*-
"""
Created on June  20  2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# bu veride başlık satırı olmadığı için header kısmını ekliyoruz
veriler = pd.read_csv('sepet.csv', header = None)

# veriyi satırlar halinde listeliyoruz
t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])
    
# Birliktelik Kural Çıkarımı / Association Rule Mining
# Apriori

from apyori import apriori
# en az 2 ürünle yapacağımız kampanya için 2li ürünleri seçiyoruz
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(kurallar))





