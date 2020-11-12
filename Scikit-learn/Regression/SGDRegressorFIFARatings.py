#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('FIFARatings.csv')
df.describe()
df.info()
print(df.shape)


# In[3]:


print(df.loc[:, 'Special'].values[:5])
print(df.loc[:,'International Reputation'].values[:5])
print(df.loc[:,'Skill Moves'].values[:5])
print(df.loc[:,'Special'].values[:5])
print(df.loc[:,'Real Face'].values[:5])
print(df.loc[:,'Jersey Number'].values[:5])
print(df.loc[:,'Joined'].values[:5])
print(df.loc[:,'Loaned From'].values[:5])
df.loc[:,'Release Clause'].values[:5]


# In[4]:


df.head()


# In[5]:


df.drop(labels=['Unnamed: 0', 'ID', 'Name', 'Photo', 'Flag', 'Club Logo', 
                'Release Clause', 'Joined', 'Loaned From', 'Jersey Number', 
                'Real Face', 'Contract Valid Until', 'Value', 'Wage'], axis=1, inplace=True)
print(df.columns)
df.drop(df.columns[range(15,41)], axis = 1, inplace=True)
print(df.columns)
df.shape


# In[6]:


df.dropna(inplace=True)
df.shape


# In[7]:


df.columns


# In[8]:


#Which Columns need to be LabelEncoded
print(df.loc[:, 'Nationality'].values[:20])
print(df.loc[:, 'Work Rate'].values[:20])
print(df.loc[:, 'Club'].values[:20])
print(df.loc[:, 'Preferred Foot'].values[:20])
print(df.loc[:, 'Body Type'].values[:20])
print(df.loc[:, 'Position'].values[:20])

le = LabelEncoder()
df['Nationality'] = le.fit_transform(df.loc[:, 'Nationality'].values)
df['Work Rate'] = le.fit_transform(df.loc[:, 'Work Rate'].values)
df['Club'] = le.fit_transform(df.loc[:, 'Club'].values)
df['Preferred Foot'] = le.fit_transform(df.loc[:, 'Preferred Foot'].values)
df['Body Type'] = le.fit_transform(df.loc[:, 'Body Type'].values)
df['Position'] = le.fit_transform(df.loc[:, 'Position'].values)

print('_________________________')

print(df.loc[:, 'Nationality'].values[:20])
print(df.loc[:, 'Work Rate'].values[:20])
print(df.loc[:, 'Club'].values[:20])
print(df.loc[:, 'Preferred Foot'].values[:20])
print(df.loc[:, 'Body Type'].values[:20])
print(df.loc[:, 'Position'].values[:20])


# In[9]:


df.columns


# In[10]:


tempDF = df[['Nationality', 'Work Rate','Club','Preferred Foot','Body Type','Position', 'Overall']]
print(tempDF.head())
df.drop(labels= ['Nationality', 'Work Rate','Club','Preferred Foot','Body Type','Position', 'Overall'], axis = 1, inplace=True)


# In[11]:


#Some more column formatting
df['Height'] = df['Height'].apply(lambda x: float(x[0]) * 12 + float(x[-1]))
df['Weight'] = df['Weight'].apply(lambda x: float(x[:-3]))


# In[12]:


#Which Columns need to be Standardized then Normalized, StandardScaler used to preserve outliers
#And MinMaxScaler includes these outliers
#Then applying PCA before recombining non-continuous data features
SS = StandardScaler()
df = SS.fit_transform(df)
mm = MinMaxScaler()
df = mm.fit_transform(df)


pca = PCA(svd_solver='randomized', n_components=5)
X_principal = pca.fit_transform(df)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = range(5) 
df = X_principal
df.head(2)


# In[13]:


df = pd.concat([pd.DataFrame(df), pd.DataFrame(tempDF)], axis=1, join='inner')
print(df.shape)
df.head()


# In[14]:


#Creating Model
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15)
sgdr = SGDRegressor(loss='epsilon_insensitive', alpha = 0.00001, epsilon = 0.0025, 
                    eta0=0.25, penalty='elasticnet', max_iter = 10000000)


# In[15]:


sgdr.fit(xtrain, ytrain)
score = sgdr.score(xtrain, ytrain)
print("R-squared:", score)
cv = RepeatedKFold(n_splits=5, n_repeats=2)
cv_score = cross_val_score(sgdr, X, y, cv=cv)
print("CV mean score: ", cv_score.mean())


# In[16]:


ypred = sgdr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




