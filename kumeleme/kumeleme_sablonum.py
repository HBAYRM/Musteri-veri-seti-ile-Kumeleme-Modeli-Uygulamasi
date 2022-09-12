# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 18:43:12 2022

@author: hbayr
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme------
#--------------------------------------------------
#2.1.veri yukleme
veriler = pd.read_csv('musteriler.csv')
print(veriler)
X=veriler.iloc[:,3:].values
Y=veriler[["Yas"]].values


#K-Means
#--------------------------------
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4,init="k-means++")
kmeans.fit(X)
y_predkmeans=kmeans.predict(X)
#görselleştirme
plt.scatter(X[y_predkmeans==0,0],X[y_predkmeans==0,1],s=100,color="black")
plt.scatter(X[y_predkmeans==1,0],X[y_predkmeans==1,1],s=100,color="yellow")
plt.scatter(X[y_predkmeans==2,0],X[y_predkmeans==2,1],s=100,color="pink")
plt.scatter(X[y_predkmeans==3,0],X[y_predkmeans==3,1],s=100,color="orange")
plt.show()
#en optimum küme sayısını hesaplama
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=(123))
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
plt.plot(range(1,11),sonuclar)
plt.show()
#----------------------------------------------------------------------------------------------------------------
#Hiyerarşik Kümeleme(HK)    
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=(4),affinity="euclidean",linkage="ward")
Y_tahmin=ac.fit_predict(X)
#görselleştirme
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,color="red")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,color="blue")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,color="green")
plt.show()
#Dendogram çizme(en iyi küme sayısını bulmaya yarar)
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show()




























