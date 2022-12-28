# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:23:22 2022

@author: Rahul
"""

import pandas as pd
import numpy as np

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\PCA\\wine.csv")
df
df.shape
df.dtypes
df['Type'].value_counts()

# Droping the first column as per the question
df1 = df.iloc[:,1:]
df1

df1.info()
df.duplicated()
df1[df1.duplicated()]

# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df2 = SS.fit_transform(df1)
df2 = pd.DataFrame(df2)

# PCA

from sklearn.decomposition import PCA
pca = PCA()
Y = pca.fit_transform(df2)

# PCA Components matrix or covariance Matrix
pca.components_
# The amount of variance that each PCA has
percentage = pca.explained_variance_ratio_
per_1 = np.cumsum(np.round(percentage,4)*100)
# Variance plot for PCA components obtained 
import matplotlib.pyplot as plt
plt.plot(per_1,color = 'magenta')

# Final Dataframe
final_df = pd.concat([df['Type'],pd.DataFrame(Y[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df

#  Visualization of PCAs
import seaborn as sns
fig = plt.figure(figsize=(16,12))
sns.scatterplot(data = final_df)

# Checking with other Clustering Algorithms
# 1. Hierarchical Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# Normalized data, Create Dendrograms
plt.figure(figsize=(10,8))
dendrograms = sch.dendrogram(sch.linkage(df2,'complete'))

H_clusters = AgglomerativeClustering(n_clusters = 3, affinity='euclidean',linkage='ward')
H_clusters

y = pd.DataFrame(H_clusters.fit_predict(df2),columns=['clustersid'])
y['clustersid'].value_counts()

# Adding Clusters to dataset
df3 = df.copy()
df3['clustersid'] = H_clusters.labels_
df3

# 2. K-Means Clustering
from sklearn.cluster import KMeans

# Use Elbow Graph to find optimum number of  clusters (K value) from K values range
# The K-means algorithm aims to choose centroids 

# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df2)
    wcss.append(kmeans.inertia_)

# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Build Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(df2)
clusters3

clusters3.labels_

# Assign clusters to the data set
K_data=df.copy()
K_data['clusters3id']=clusters3.labels_
K_data

K_data['clusters3id'].value_counts()






























