#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.svm import SVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd


# In[2]:


#Just for runtime sake only using 10k samples
df = pd.read_csv('hotel_bookings.csv').loc[:25000, :]
print(df.describe())
print(df.info())
df.shape


# In[3]:


df.columns = [feature.strip() for feature in df.columns]
print(df.columns)
df.head()


# In[4]:


colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)


# In[5]:


df.drop(df.columns[-9:-7], axis=1, inplace=True)
colNA = []
for col in df.columns:
    colNA.append(df[col].isna().sum())
print(colNA)
df.head()


# In[6]:


naNames = [df.columns[index] for index in range(len(colNA)) if colNA[index] > 0]
print(naNames)
print(df['children'].isna().sum())
print(df['country'].isna().sum())
df.fillna(value=0.0, inplace=True)
df.drop(labels= ['country', 'reservation_status_date'], axis=1, inplace=True)
print(df.columns)
df.shape


# In[7]:


#Which Columns need to be LabelEncoded
print(pd.unique(df['hotel']))
print(len(pd.unique(df['stays_in_weekend_nights'])))
print(len(pd.unique(df['adr'])))
print(len(pd.unique(df['required_car_parking_spaces'])))
print(pd.unique(df['distribution_channel']))
print(len(pd.unique(df['previous_cancellations'])))
print(len(pd.unique(df['previous_bookings_not_canceled'])))
print(len(pd.unique(df['booking_changes'])))
print(pd.unique(df['meal']))



le = LabelEncoder()
needEncoding = ['hotel', 'is_canceled','arrival_date_year','arrival_date_month','arrival_date_week_number',
                'market_segment','distribution_channel','is_repeated_guest','reserved_room_type','assigned_room_type',
                'deposit_type','customer_type','reservation_status', 'arrival_date_day_of_month', 'meal']
for name in needEncoding:
    df[name] = le.fit_transform(df[name].values)
df.head()


# In[8]:


df['Total Occupants'] = df['adults'] + df['children'] + df['babies']
df.drop(labels=['adults', 'children', 'babies'], axis = 1, inplace=True)
tempDF = df[[col for col in df.columns if col not in needEncoding]]
df.drop(labels=[col for col in df.columns if col not in needEncoding], axis = 1, inplace=True)
total = tempDF[['Total Occupants']]
df = pd.concat([df, total], axis = 1, join='inner')
tempDF.drop(labels=['Total Occupants'], axis=1, inplace=True)
print(df['Total Occupants'].values[:10])
print(tempDF.columns)
print(df.columns)


# In[9]:


#Which Columns need to be Standardized then Normalized, RobustScaler since want resistance to outliers
#Using normalize to be resistant to outliers
#Then applying KernelApproximation before recombining non-continuous data features
robust = RobustScaler()
tempDF = robust.fit_transform(tempDF)
tempDF = normalize(tempDF)

rbf = RBFSampler(n_components = 10, gamma = 1)
tempDF = pd.DataFrame(rbf.fit_transform(tempDF))
tempDF.head()


# In[10]:


df = pd.concat([tempDF, df], axis=1, join='inner')
print(df.shape)
df.head()


# In[11]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15)


# In[12]:


cv = RepeatedKFold(n_splits = 3, n_repeats = 1)
SVR = SVR(kernel= 'rbf', gamma=None, C=None)
param_grid = dict()
param_grid['C'] = [0.01, 0.1, 1]
param_grid['gamma'] = [0.01, 0.1, 1]
model = GridSearchCV(SVR, param_grid = param_grid, cv=cv, n_jobs=-1)
model.fit(xtrain, ytrain)


# In[13]:


ypred = model.predict(xtest)
r2 = r2_score(ytest, ypred)
print('R2 Score: ', r2)
score = model.score(xtrain, ytrain)
print("R-squared:", score)


# In[14]:


mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))

