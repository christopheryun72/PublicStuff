#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('birds_united_states.csv')
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


df.drop(df.columns[[i for i in range(len(df.columns)) if colNA[i] > 2000]], axis=1, inplace=True)
print(df.shape)
colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)


# In[6]:


df.dropna(inplace=True)
print(df.shape)
colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)
print(df.columns)
df.head()


# In[7]:


print(len(pd.unique(df['gen'].values)))
print(len(pd.unique(df['rec'].values)))
print(len(pd.unique(df['type'].values)))
print(pd.unique(df['q'].values))


# In[8]:


df.drop(labels=['id', 'url', 'file', 'file-name', 'sono', 'lic', 
                'time', 'date', 'uploaded', 'also'], axis=1, inplace=True)
print(df.shape)
print(df.columns)
df.head()


# In[9]:


df['length'].values[:30]


# In[10]:


#Column Type Modifications
df['Class'] = df['q']
df.drop(labels=['q'], axis =1, inplace=True)
df['length'] = df['length'].apply(lambda x: (float(str(x)[0]) * 100.0) + float(str(x)[-2:]))
df.head()


# In[11]:


tempDF = df[['lat', 'lng', 'alt', 'length']]
df.drop(labels = tempDF.columns, axis=1, inplace=True)
print(tempDF.head())
df.head()


# In[12]:


print(tempDF.shape)
tempDF['alt2'] = pd.DataFrame(tempDF['alt'].apply(pd.to_numeric, errors='ignore')).values
#print(pd.unique(tempDF['alt2']))
tempDF['alt3'] = pd.DataFrame(tempDF['alt2'].apply(lambda x: True if not isinstance(x, str) else np.nan)).values
tempDF.dropna(inplace=True)
print(tempDF.shape)
tempDF.head()


# In[13]:


tempDF['alt'] = tempDF['alt2']
tempDF.drop(labels=['alt2', 'alt3'], axis=1, inplace=True)
tempDF.head()


# In[14]:


#Which Columns need to be Standardized then Normalized, RobustScaler since don't want influence of outliers
#Using normalize to not allow influence of outliers
#Then applying RBFSampler before recombining non-continuous data features
robust = RobustScaler()
tempDF = robust.fit_transform(tempDF)
tempDF = normalize(tempDF)

rbf = RBFSampler(n_components = 2, gamma = 1.5)
tempDF = pd.DataFrame(rbf.fit_transform(tempDF))
tempDF.head()


# In[15]:


#LabelEncoding
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col].values)
df.head()


# In[16]:


df = pd.concat([tempDF, df], axis=1, join='inner')
df.head()


# In[17]:


cv = RepeatedKFold(n_splits=5, n_repeats=2)
param_grid = dict()
param_grid['n_clusters'] = list(np.arange(8, 12))
param_grid['n_init'] = list(np.arange(2, 5))
param_grid['batch_size'] = [175, 200, 225]
mbkm = MiniBatchKMeans(init='k-means++', init_size=None) 
model = GridSearchCV(mbkm, param_grid=param_grid, cv=cv, n_jobs=-1)


# In[18]:


model.fit(df)
model.best_estimator_


# In[20]:


ypred = model.predict(df)
sil = silhouette_score(df, ypred)
print('Silhouette Score: ', sil)


# In[ ]:




