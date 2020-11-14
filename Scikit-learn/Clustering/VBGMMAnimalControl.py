#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('BR_Animal_Control_Calls.csv')
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


df.dropna(inplace=True, thresh = df.shape[1] - 1)
colNA =[]
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)
df.head()


# In[6]:


df.drop(df.columns[[-6]], axis=1, inplace=True)
print(df.shape)
colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)
df.head()


# In[7]:


df.drop(labels=['file_number', 'incident_date', 'location', 'zip_code', 'municipality'], axis=1, inplace=True)


# In[8]:


df['request_type'] = df['request_type'].apply(lambda x: 'maybe' if not isinstance(x, str) else x)
df = df.sample(frac=0.5, replace=True, random_state=1)
print(df.shape)
df.head()


# In[9]:


#No Standardizing nor Normalizing needed because lat and long are on the same scale
#LabelEncoding
le = LabelEncoder()
for col in df.columns[:-2]:
    df[col] = le.fit_transform(df[col].values)
df.head()


# In[10]:


cv = RepeatedKFold(n_splits=4, n_repeats=2)
param_grid = dict()
param_grid['covariance_type'] = ['full', 'drag', 'tied', 'spherical']
param_grid['n_components'] = list(np.arange(1, 4))
vbgmm = BayesianGaussianMixture(weight_concentration_prior = 0.5, n_init=2, init_params='kmeans')
model = GridSearchCV(vbgmm, param_grid=param_grid, cv=cv, n_jobs=-1)


# In[11]:


model.fit(df)
model.best_estimator_


# In[12]:


ypred = model.predict(df)
sil = silhouette_score(df, ypred)
print('Silhouette Score: ', sil)

