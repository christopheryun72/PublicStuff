#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('IBMAttrition.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df.columns)
df.head()


# In[4]:


df.columnns = [feature.strip() for feature in df.columns]
colNA = []
for feature in df.columns:
    colNA.append(df[feature].isna().sum())
print('Columns with NA/NaNs:', colNA)


# In[5]:


print(pd.unique(df['EnvironmentSatisfaction'].values))
print(pd.unique(df['JobInvolvement'].values))
print(pd.unique(df['JobLevel'].values))
print(pd.unique(df['JobRole'].values))
print(pd.unique(df['JobSatisfaction'].values))
print(pd.unique(df['PerformanceRating'].values))
print(pd.unique(df['RelationshipSatisfaction'].values))
print(pd.unique(df['StandardHours'].values))
print(pd.unique(df['StockOptionLevel'].values))
print(pd.unique(df['WorkLifeBalance'].values))

#LabelEncoding
df.drop(labels=['StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
discrete = ['BusinessTravel', 'Department', 'EducationField', 'EnvironmentSatisfaction', 'JobInvolvement',
             'JobLevel', 'Gender', 'JobRole', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
             'StockOptionLevel', 'WorkLifeBalance', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
tempDF = df[discrete]
df.drop(labels=discrete, axis=1, inplace=True)
print(df.columns)
df.head()


# In[6]:


print(tempDF.columns)
tempDF.head()


# In[7]:


#LabelEncoding
le = LabelEncoder()
for feature in tempDF.columns:
    tempDF[feature] = le.fit_transform(tempDF[feature].values)
tempDF.head()


# In[8]:


#Which Columns need to be Standardized then Normalized, RobustScaler since want resistance to outliers
#Using normalize to be resistant to outliers
#Then applying PCA before recombining non-continuous data features
robust = RobustScaler()
df = robust.fit_transform(df)
df = normalize(df)

pca = PCA(svd_solver='randomized', n_components = 10)
df = pd.DataFrame(pca.fit_transform(df))
df.head()


# In[9]:


df = pd.concat([df, tempDF], axis = 1, join='inner')
print(df.columns)
df.head()


# In[10]:


X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15)


# In[11]:


cv = RepeatedKFold(n_splits=4, n_repeats=3)
param_grid = dict()
param_grid['kernel'] = ['poly', 'rbf']
param_grid['degree'] = [3, 4, 5]
param_grid['C'] = list(np.arange(0.5, 2, 0.25))
svc = SVC(max_iter = 10000, gamma = 'scale')
model = GridSearchCV(svc, param_grid= param_grid, cv=cv)


# In[12]:


model.fit(xtrain, ytrain)
model.best_estimator_


# In[13]:


ypred = model.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)
mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))
accuracy = accuracy_score(ytest, ypred)
print('Accuracy: ', accuracy)


# In[ ]:




