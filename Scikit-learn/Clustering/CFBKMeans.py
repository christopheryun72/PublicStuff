#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('CFBeattendance.csv', encoding= 'unicode_escape')
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


print(pd.unique(df['TMAX'].values[:15]))


# In[6]:


df.drop(labels=['Date', 'Site', 'TV', 'Result', 'Stadium Capacity', 
                'TMAX', 'TMIN', 'Year', 'Day', 'Opponent'], axis=1, inplace=True)
print(df.columns)
df.head()


# In[7]:


print(len(pd.unique(df['Rank'].values)))
print(len(pd.unique(df['Team'].values)))
print(pd.unique(df['New Coach'].values))
print(pd.unique(df['Tailgating'].values))
print(len(pd.unique(df['PRCP'].values)))
print(pd.unique(df['SNOW'].values))
print(pd.unique(df['SNWD'].values))
print(pd.unique(df['Conference'].values))


# In[8]:


#Formatting Time Feature
df['Time'] = df['Time'].apply(lambda x: (int(str(x)[:-6]) * 60) + int(str(x)[-5:-2]) + 720 
                              if str(x)[-2:] == 'PM' else (int(str(x)[:-6]) * 60) + int(str(x)[-5:-2]))
print(df['Time'].values[:5])


# In[9]:


tempDF = df[['Team', 'Rank', 'New Coach', 'Tailgating', 'Opponent_Rank', 'Conference', 'Month']]
df.drop(labels=tempDF.columns, axis=1, inplace=True)
print(tempDF.head())
df.head()


# In[10]:


#LabelEncoding
le = LabelEncoder()
for col in tempDF.columns:
    tempDF[col] = le.fit_transform(tempDF[col].values)
tempDF.head()


# In[11]:


#Which Columns need to be Standardized then Normalized, RobustScaler since don't want influence of outliers
#Using normalize to not allow influence of outliers
#Then applying Isomap before recombining non-continuous data features
robust = RobustScaler()
df = robust.fit_transform(df)
df = normalize(df)

iso = Isomap(n_neighbors = 6, n_components=3, path_method='auto', neighbors_algorithm = 'kd_tree')
df = pd.DataFrame(iso.fit_transform(df))
df.head()


# In[12]:


df = pd.concat([pd.DataFrame(tempDF), df], axis=1, join='inner')
df.head()


# In[13]:


cv = RepeatedKFold(n_splits=5, n_repeats=2)
param_grid = dict()
param_grid['n_clusters'] = list(np.arange(8, 12))
param_grid['n_init'] = list(np.arange(8, 12))
param_grid['algorithm'] = ['full', 'elkan']
km = KMeans(init='k-means++', max_iter=500)
model = GridSearchCV(km, param_grid=param_grid, cv=cv, n_jobs=-1)


# In[14]:


model.fit(df)
model.best_estimator_


# In[15]:


ypred = model.predict(df)
sil = silhouette_score(df, ypred)
print('Silhouette Score: ', sil)

