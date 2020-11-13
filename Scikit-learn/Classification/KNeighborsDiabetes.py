#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('diabetes.csv')
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


y = df['Outcome'].values
df.drop(labels=['Outcome'], axis =1, inplace=True)
print(df.columns)
df.head()


# In[6]:


#Which Columns need to be Standardized then Normalized, StandardScaler since want influence of outliers
#Using MinMaxScaler to allow influence of outliers
#Then applying PCA before recombining non-continuous data features
SS = StandardScaler()
df = SS.fit_transform(df)
mm = MinMaxScaler()
df = mm.fit_transform(df)

pca = PCA(svd_solver = 'full', n_components=4)
df = pd.DataFrame(pca.fit_transform(df))
print(df.shape)
df.head()


# In[7]:


X = df.values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)


# In[8]:


cv = RepeatedKFold(n_splits=3, n_repeats=2)
param_grid = dict()
param_grid['leaf_size'] = list(np.arange(10, 40, 5))
param_grid['n_neighbors'] = list(np.arange(2, 7))
knn = KNeighborsClassifier(algorithm = 'kd_tree', metric='minkowski')
model = GridSearchCV(knn, param_grid=param_grid, cv=cv)


# In[9]:


model.fit(xtrain, ytrain)
model.best_estimator_


# In[10]:


ypred = model.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)
mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))
accuracy = accuracy_score(ytest, ypred)
print("Accuracy: ", accuracy)

