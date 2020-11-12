#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.linear_model import LassoCV
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.kernel_approximation import Nystroem
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('2018-personality-data.csv')
df.describe()
df.info()
print(df.shape)


# In[3]:


df.columns = [feature.strip() for feature in df.columns]
print(df.loc[:, 'movie_1'].values[:5])
print(df.loc[:, 'predicted_rating_1'].values[:5])
print(df.loc[:, 'openness'].values[:5])
print(df.loc[:, 'enjoy_watching'].values[:5])


# In[4]:


df.head()


# In[5]:


df.drop(df.columns[range(8, 33)], axis = 1, inplace=True)
df.drop(labels=['userid'], axis = 1, inplace=True)
print(df.columns)
df.shape


# In[6]:


colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)
df.dropna(inplace=True)
print(df.head())
df.shape


# In[7]:


#Which Columns need to be LabelEncoded
le = LabelEncoder()
df['assigned condition'] = le.fit_transform(df.loc[:, 'assigned condition'].values)
df['assigned metric'] = le.fit_transform(df.loc[:, 'assigned metric'].values)
print(df.loc[:, 'assigned condition'].values)
print(df.loc[:, 'assigned metric'].values)


# In[8]:


tempDF = df[['assigned condition', 'assigned metric', 'enjoy_watching']]
print(tempDF.head())
df.drop(labels=['assigned condition', 'assigned metric', 'enjoy_watching'], axis = 1, inplace = True)
df.head()


# In[9]:


#Which Columns need to be Standardized then Normalized, RobustScaler since want resistance to outliers
#Using normalize to be resistant to outliers
#Then applying Nystroem before recombining non-continuous data features
robust = RobustScaler()
df = robust.fit_transform(df)
df = normalize(df)

#In retrospect, should not have used Kernel Approximation b/c not enough sample points
#Should have tried Isomap
nys_feature_map = Nystroem(kernel = 'polynomial', degree=3, n_components=2, gamma = 0.1)
df = pd.DataFrame(nys_feature_map.fit_transform(df))
df.head()


# In[10]:


df = pd.concat([df, pd.DataFrame(tempDF)], axis=1, join='inner')
print(df.shape)
df.head()


# In[11]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15)


# In[12]:


cv = RepeatedKFold(n_splits = 5, n_repeats=3)
lasso = LassoCV(alphas=None, cv = cv, max_iter = 100000)
lasso.fit(xtrain, ytrain)


# In[13]:


ypred = lasso.predict(xtest)
r2 = r2_score(ytest, ypred)
print('R2 Score: ', r2)
score = lasso.score(xtrain, ytrain)
print("R-squared:", score)


# In[14]:


mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))


# In[ ]:




