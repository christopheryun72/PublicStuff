#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('TelcoCustomerChurn.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df.columns)
df.head()


# In[4]:


df.columns = [feature.strip() for feature in df.columns]
df.drop(labels=['customerID'], axis=1, inplace=True)
unique = []
for col in df.columns:
    unique.append(len(pd.unique(df[col].values)))
print(unique)


# In[5]:


tempDF = pd.concat([df.iloc[:, -3:-1], df.iloc[:, 4]], axis=1)
df.drop(labels = tempDF.columns, axis=1, inplace=True)
print(tempDF.head())
df.head()


# In[6]:


#LabelEncoding
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col].values)
df.head()


# In[7]:


def delete(x):
    try: 
        return float(x)
    except:
        return np.nan
tempDF['TotalCharges'] = tempDF['TotalCharges'].apply(delete)
tempDF.dropna(inplace=True)


# In[8]:


#Which Columns need to be Standardized then Normalized, RobustScaler since don't want influence of outliers
#Using normalize to not allow influence of outliers
#Then applying RBFSampler before recombining non-continuous data features
robust = RobustScaler()
tempDF = robust.fit_transform(tempDF)
tempDF = normalize(tempDF)

pca = PCA(n_components=2, svd_solver='full')
tempDF = pd.DataFrame(pca.fit_transform(tempDF))
tempDF.head()


# In[9]:


df = pd.concat([tempDF, df], axis=1, join='inner')
df.head()


# In[10]:


cv = RepeatedKFold(n_splits=4, n_repeats=2)
param_grid = dict()
param_grid['init_params'] = ['kmeans', 'random']
param_grid['n_components'] = list(np.arange(2, 8))
param_grid['n_init'] = list(np.arange(2, 8))
gmm = GaussianMixture(covariance_type='full', max_iter=300)
model = GridSearchCV(gmm, param_grid=param_grid, cv=cv, n_jobs=-1)


# In[11]:


model.fit(df)
model.best_estimator_


# In[12]:


ypred = model.predict(df)
sil = silhouette_score(df, ypred)
print('Silhouette Score: ', sil)


# In[ ]:




