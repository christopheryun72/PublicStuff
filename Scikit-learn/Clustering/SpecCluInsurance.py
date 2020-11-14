#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('insurance.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df.columns)
df.head()


# In[4]:


#Data is already formatted and all columns are useful
tempDF = df[['sex', 'smoker', 'region']]
df.drop(labels = tempDF.columns, axis=1, inplace=True)
print(tempDF.head())
df.head()


# In[5]:


#LabelEncoding
le = LabelEncoder()
for col in tempDF.columns:
    tempDF[col] = le.fit_transform(tempDF[col].values)
tempDF.head()


# In[6]:


#Which Columns need to be Standardized then Normalized, StandardScaler since want influence of outliers
#Using MinMaxScaler to allow influence of outliers
#Then applying SpectralEmbedding before recombining non-continuous data features
SS = StandardScaler()
df = SS.fit_transform(df)
mm = MinMaxScaler()
df = mm.fit_transform(df)

se = SpectralEmbedding(n_components = 2, affinity='rbf', gamma=0.8)
df = pd.DataFrame(se.fit_transform(df))
df.head()


# In[7]:


df = pd.concat([tempDF, df], axis=1, join='inner')
df.head()


# In[8]:


cv = RepeatedKFold(n_splits=6, n_repeats=3)
param_grid = dict()
param_grid['n_clusters'] = list(np.arange(1, 5, 1)) 
param_grid['gamma'] = list(np.arange(0.6, 1.6, .2))
param_grid['degree'] = list(np.arange(2, 6, 1))
sc = SpectralClustering(assign_labels = 'kmeans', affinity='rbf', degree=4, gamma=.8)
#model = GridSearchCV(mbkm, param_grid=param_grid, cv=cv, n_jobs=-1)


# In[ ]:


"""
model.fit(df)
model.best_estimator
"""


# In[9]:


ypred = sc.fit_predict(df)
sil = silhouette_score(df, ypred)
print('Silhouette Score: ', sil)


# In[ ]:




