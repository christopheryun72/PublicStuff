#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.linear_model import ElasticNetCV
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('winequality-red.csv')
df.describe()
df.info()
print(df.shape)


# In[3]:


df.columns = [feature.strip() for feature in df.columns]
print(df.loc[:, 'fixed acidity'].values[:5])
print(df.loc[:, 'density'].values[:5])
print(df.loc[:, 'quality'].values[:5])


# In[4]:


df.head()


# In[5]:


colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)
df.dropna(inplace=True)
print(df.shape)
#df.head()


# In[6]:


#Which Columns need to be Standardized then Normalized, StandardScaler since don't want resistance to outliers
#Using MinMaxScaler to be influenced by outliers
#Then applying Isomap before training
y = df.iloc[:, -1].values
SS = StandardScaler()
df = SS.fit_transform(df[:-1])
mm = MinMaxScaler()
df = mm.fit_transform(df[:-1])

isom = Isomap(n_neighbors=3, n_components=4, path_method = 'auto', neighbors_algorithm = 'kd_tree')
df = pd.DataFrame(isom.fit_transform(df[:-1]))
y = pd.DataFrame(y)
y.columns = ['Score']
df = pd.concat([df, y], axis=1, join='inner')
print(df.shape)
df.head()


# In[7]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15)


# In[8]:


cv = RepeatedKFold(n_splits=5, n_repeats = 2)
enet = ElasticNetCV(alphas=None, cv=cv, max_iter=100000, l1_ratio = 0.3)
enet.fit(xtrain, ytrain)


# In[9]:


ypred = enet.predict(xtest)
r2 = r2_score(ytest, ypred)
print('R2 Score: ', r2)
score = enet.score(xtrain, ytrain)
print('R-Squared: ', score)


# In[10]:


mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))


# In[ ]:




