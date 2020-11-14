#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('ShelterDogs.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df.columns)
df.head()


# In[4]:


df.columns = [feature.strip() for feature in df.columns]
colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)


# In[5]:


df.drop(df.columns[[i for i in range(len(df.columns)) if colNA[i] > 1305]], axis=1, inplace=True)
print(df.shape)
colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)


# In[6]:


naCols = []
for i in range(len(df.columns)):
    if (colNA[i] != 0):
        naCols.append(df.columns[i])
print(naCols)


# In[7]:


for col in naCols:
    print(pd.unique(df[col]))


# In[8]:


df.drop(labels=['name'], axis=1, inplace=True)
naCols = naCols[1:]
for feature in naCols:
    df[feature] = df[feature].apply(lambda x: 'maybe' if not isinstance(x, str) else x)
for col in naCols:
    print(pd.unique(df[col]))


# In[9]:


df.head()


# In[10]:


df.drop(labels=['ID', 'date_found', 'adoptable_from', 'posted'], axis=1, inplace=True)
df.head()


# In[11]:


#No Standardizing nor Normalizing needed with only one continuous feature
#LabelEncoding
le = LabelEncoder()
for col in df.columns[1:]:
    df[col] = le.fit_transform(df[col].values)
df.head()


# In[12]:


cv = RepeatedKFold(n_splits=6, n_repeats=2)
bandwidth = estimate_bandwidth(df, quantile=0.5, n_samples=df.shape[0])
ms = MeanShift(bin_seeding=True, cluster_all=True, bandwidth=bandwidth)


# In[13]:


ms.fit(df)


# In[14]:


ypred = ms.predict(df)
sil = silhouette_score(df, ypred)
print('Silhouette Score: ', sil)

