# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('scores.csv')
data = data.drop(columns=['School ID', 'Street Address', 'State', 'Zip Code', 'Latitude', 'Longitude', 'Phone Number'], axis=1)
data

X = data.iloc[:, 1:11].values
print(X)
print(type(X))

imrAnother = SimpleImputer(missing_values=np.nan, strategy='mean')
imrAnother = imrAnother.fit(X[:, 5:6])
X[:, 5:6] = imrAnother.transform(X[:, 5:6])
print(X[0])
print(X[1])

for i in range(len(X)):
  for y in range(len(X[i])):
    try:
      if y == 3 or y == 4:
        X[i][y] = X[i][y][:-3]
        mins = X[i][y].split(":")[1]
        hrs = X[i][y].split(":")[0]
        X[i][y] = int(hrs) * 60 + int(mins)
      if y == 6 or y == 7 or y == 8 or y == 9:
        X[i][y] = float(X[i][y][:-1])
    except:
      pass
print(X[0])
print(X[1])
print(type(X))

imrSecond = SimpleImputer(missing_values=np.nan, strategy='mean')
imrSecond = imrSecond.fit(X[:, 3:5])
X[:, 3:5] = imrSecond.transform(X[:, 3:5])
imrThird = SimpleImputer(missing_values=np.nan, strategy='mean')
imrThird = imrSecond.fit(X[:, 6:10])
X[:, 6:10] = imrSecond.transform(X[:, 6:10])
print(X[0])
print(X[1])

y = data.iloc[:, -4:-1].values
tempY = []
for i in range(len(y)):
  sum = 0
  for w in range(len(y[i])):
    sum = sum + float(y[i][w])
  tempY.append(sum)
y = np.asarray(tempY)
print(y)

y = y.reshape(-1, 1)
imrY = SimpleImputer(missing_values=np.nan, strategy='mean')
imrY = imrY.fit(y)
y = imrY.transform(y).flatten()
y = np.around(y, decimals=0)
print(y)

print(X[0:10])

le = LabelEncoder()
for i in range(0,3):
  X[:, i] = le.fit_transform(X[:, i])
X = X.astype(float)
X = np.around(X, decimals=0)
print(X[0:10])
print(y[0:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)

annModel = tf.keras.models.Sequential()
annModel.add(tf.keras.layers.Dense(units=1024, activation='relu'))
annModel.add(tf.keras.layers.Dropout(rate=0.1))
annModel.add(tf.keras.layers.Dense(units=1024, activation='relu'))
annModel.add(tf.keras.layers.Dropout(rate=0.1))
annModel.add(tf.keras.layers.Dense(units=1024, activation='relu'))
annModel.add(tf.keras.layers.Dropout(rate=0.1))
annModel.add(tf.keras.layers.Dense(units=1024, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 1))
#Tried Leaky ReLU, Didn't Help Much. Regularizers made loss worse, weird...

annModel.compile(optimizer='adam', loss='mean_squared_error')
annModel.fit(X_train, y_train, batch_size=8, epochs=1500)

#Testing if Overfitting/Underfitting
evalTest = annModel.evaluate(X_test, y_test, batch_size=16)
print(evalTest)
evalTrain = annModel.evaluate(X_train, y_train, batch_size=16)
print(evalTrain)

#Confirming Overfitting via K-Fold Cross-Validation
def get_score(model, X_train, X_test, y_train, y_test):
  model.fit(X_train, y_train)
  return model.evaluate(X_test, y_test, batch_size =16)


kf = KFold(n_splits = 10)
scores = []
for train_index, test_index in kf.split(X):
  X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
  #If wanting to test on multiple, different models, line below but with diff model name and obviously different container
  scores.append(get_score(annModel,X_train, X_test, y_train, y_test))
  #print(train_index, test_index)
scores
