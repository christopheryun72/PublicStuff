#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('InstrumentalRatio.csv')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df.head())
df.columns


# In[4]:


df.columns = [feature.strip() for feature in df.columns]
df.drop(labels=['artists', 'id','name','release_date'], axis=1, inplace=True)
print(type(df.columns))
df.head()


# In[5]:


print(pd.unique(df['key']))
print(pd.unique(df['mode']))
print(pd.unique(df['popularity']))
print(pd.unique(df['explicit']))
colNA = []
for feature in df.columns:
    colNA.append(df[feature].isna().sum())
print('Columns with NA/NaNs:', colNA)


# In[6]:


tempDF = df[['explicit', 'key', 'mode']]
df.drop(labels = [feature for feature in df.columns if feature in tempDF.columns], axis = 1, inplace=True)
print(df.shape)
print(df.head())
tempDF.head()


# In[7]:


#Label Encoding Just For Insurance
le = LabelEncoder()
for feature in tempDF.columns:
    le.fit_transform(tempDF[feature].values)
tempDF.head()


# In[8]:


#Which Columns need to be Standardized then Normalized, RobustScaler since want resistance to outliers
#Using normalize to be resistant to outliers
#Then applying RBFSampler before recombining non-continuous data features
robust = RobustScaler()
df = robust.fit_transform(df)
df = normalize(df)

rbf = RBFSampler(n_components=7, gamma=2.5)
df = pd.DataFrame(rbf.fit_transform(df))
df.head()


# In[9]:


df = pd.concat([df, tempDF], axis=1, join='inner')
df['Ratio'] = df['key']
df.drop(labels=['key'], axis=1, inplace=True)
df['Ratio'] = df['Ratio'].apply(lambda x: round(x/5))
print(df.columns)
df.head()


# In[10]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15)


# In[11]:


cv = RepeatedKFold(n_splits=6, n_repeats=3)
param_grid = dict()
param_grid['alpha'] = [0.0001, 0.001, 0.01]
param_grid['l1_ratio'] = list(np.arange(0.2, 0.6, 0.05))
sgd = SGDClassifier(penalty='elasticnet', max_iter=100000, epsilon=0.001, learning_rate='optimal', loss='log', n_jobs=-1)
model = GridSearchCV(sgd, param_grid = param_grid, cv = cv, n_jobs=-1, )
model.fit(xtrain, ytrain)


# In[12]:


print(model.best_estimator_)


# In[13]:


ypred = model.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)
mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))

