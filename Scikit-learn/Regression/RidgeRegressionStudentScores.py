#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.linear_model import RidgeCV
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('StudentsPerformance.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


df.columns = [feature.strip() for feature in df.columns]
print(df.loc[:, 'gender'].values[:5])
print(df.loc[:, 'test preparation course'].values[:5])
print(df.loc[:, 'math score'].values[:5])


# In[4]:


df.head()


# In[5]:


colNA = []
for col in df.columns: 
    colNA.append(df[col].isna().sum())
print(colNA)


# In[6]:


df['Cummulative Score'] = df['reading score'] + df['math score'] + df['writing score']
df['Cummulative Score'].values[:5]


# In[7]:


df.drop(df.columns[5:8], axis=1, inplace=True)
df.head()


# In[8]:


#Which Columns need to be LabelEncoded
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'].values)
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'].values)
df['parental level of education'] = le.fit_transform(df['parental level of education'].values)
df['lunch'] = le.fit_transform(df['lunch'].values)
df['test preparation course'] = le.fit_transform(df['test preparation course'].values)
df.head()


# In[9]:


#No standardizing nor normalization needed/possible
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15)


# In[10]:


cv = RepeatedKFold(n_splits=7, n_repeats = 2)
ridge = RidgeCV(alphas= np.arange(0, 5, 0.01), cv = cv, scoring='r2')
ridge.fit(xtrain, ytrain)


# In[11]:


ypred = ridge.predict(xtest)
r2 = r2_score(ytest, ypred)
print('R2 Score: ', r2)
score = ridge.score(xtrain, ytrain)
print('R Squared: ', score)


# In[12]:


mse = mean_squared_error(ytest, ypred)
print('Mean Squared Error: ', mse)
print("RMSE: ", mse**(1/2.0))


# In[ ]:




