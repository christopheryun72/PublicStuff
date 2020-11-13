#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df.shape)
df.head()


# In[4]:


df.columns = [feature.strip() for feature in df.columns]
print(pd.unique(df['anaemia'].values))
print(pd.unique(df['diabetes'].values))
print(pd.unique(df['time'].values))
print(pd.unique(df['DEATH_EVENT'].values))
colNA = []
for feature in df.columns:
    colNA.append(df[feature].isna().sum())
print('Columns with NA/NaNs:', colNA)


# In[5]:


tempDF = df[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']]
df.drop(labels = [feature for feature in df.columns if feature in tempDF], axis=1, inplace=True)
print(df.shape)
print(df.head())
tempDF.head()


# In[6]:


#Which Columns need to be Standardized then Normalized, StandardScaler since want influence of outliers
#Using MinMaxScaler to allow influence of outliers
#Then applying SpectralEmbedding before recombining non-continuous data features
SS = StandardScaler()
df = SS.fit_transform(df)
mm = MinMaxScaler()
df = mm.fit_transform(df)

se = SpectralEmbedding(n_components = 3, affinity='nearest_neighbors', n_neighbors=6)
df = pd.DataFrame(se.fit_transform(df))
df.head()


# In[7]:


df = pd.concat([df, tempDF], axis=1, join='inner')
print(df.shape)
df.head()


# In[8]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)


# In[9]:


cv = RepeatedKFold(n_splits=8, n_repeats=3)
param_grid = dict()
param_grid['penalty'] = ['l1', 'l2']
param_grid['loss'] = ['hinge', 'squared_hinge']
param_grid['C'] = list(np.arange(0, 10, 0.25))
lsvc = LinearSVC(max_iter=100000, multi_class='ovr')
model = GridSearchCV(lsvc, cv=cv, param_grid=param_grid, n_jobs=-1)
model.fit(xtrain, ytrain)


# In[10]:


model.best_estimator_


# In[11]:


ypred = model.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)
mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))

