# -*- coding: utf-8 -*-
"""
Created on June  20  2022

@author: Yunus GÜMÜŞSOY
"""

#1.kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 10 reklamın tıklanma sayısını içeren veriyi yüklüyoruz
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

"""
# 1. Random Seçim
# 10 reklamı random olarak 10,000 kere seçeceğiz 
import random

# seçim sayısı
N = 10000

# reklam sayısı
d = 10 

toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()
# bu çalıştırmada 1215 çıktı, yani rasgele seçmek yerine sadece ad1, ad2, ad5 veya ad8 i seçmek daha mantıklı


# burada önemli olan bir husus var: 
# random olarak reklam seçiminde, seçim sonucunda her reklamın toplam seçilme sayısı farklı olmalı
# datamıza göre, ad1 1703, ad2 1295, ad3 728, ad4 1196, ad5 2695, ad6 126, ad7 1112, ad8 2091, ad9 952, ad10 489

"""
"""
# Upper Confidence Bound UCB - Üst Güven Sınırı
import math

N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
#Ri(n)
oduller = [0] * d #ilk basta butun ilanların odulu 0
#Ni(n)
tiklamalar = [0] * d #o ana kadarki tıklamalar
toplam = 0 # toplam odul
secilenler = []
for n in range(1,N):
    ad = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0,d): # max ucb si olan reklamı bulmak için
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: #max'tan büyük bir ucb çıktı
            max_ucb = ucb
            ad = i          
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+ 1
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    oduller[ad] = oduller[ad]+ odul
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()

# Toplam ödülü bu sefer 2231 bulduk, rasgele seçimde 1215 di. Dolayısıyla bu algoritma rasgele seçime göre mantıklı
"""

# Thompson Sampling - Thompson Örneklemesi
import random

N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var

toplam = 0 # toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1,N):
    ad = 0 #seçilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate ( birler[i] + 1 , sifirlar[i] +1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    if odul == 1:
        birler[ad] = birler[ad]+1
    else :
        sifirlar[ad] = sifirlar[ad] + 1
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()

#Toplam ödülü bu sefer 2595 bulduk, rasgele seçimde 1215, UCB de 2231 di. Dolayısıyla bu algoritma daha başarılı