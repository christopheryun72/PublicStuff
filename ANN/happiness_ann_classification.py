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

testing = keras.utils.to_categorical(np.random.randint(10, size=(1000,1)), num_classes=10)

data = pd.read_csv('Kag_happiness_indicators.csv')
X = data.loc[:, list(data.columns[1:14]) + list(data.columns[15:17]) + list(data.columns[18:])].values
y = data.iloc[:, 17].values
print(X)
print(y)
print(X[5][:10])

le = LabelEncoder()
for i in range(1, 31):
  if i == 16 or i == 18 or i == 19:
    continue
  X[:, i] = le.fit_transform(X[:, i].astype(str))
print(X)
print(y)
print(X[0, 0:32])
X = X.astype(float)
y = y.astype(float)
print(X)
print(y)
print(X[0, 0:32])

#Checking for NaN Values Leftover
print(np.isnan(X).any())
output = np.argwhere(np.isnan(X))
output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

annModel = tf.keras.models.Sequential()
#annModel.add(tf.keras.layers.Dense(units=256, activation='relu'))
annModel.add(tf.keras.layers.Dense(units=256, activation='relu', 
                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=regularizers.l2(1e-4),
                                   activity_regularizer=regularizers.l2(1e-5)))
annModel.add(tf.keras.layers.LeakyReLU(alpha=0.05))
annModel.add(tf.keras.layers.Dropout(rate=0.3))
#annModel.add(tf.keras.layers.Dense(units=256, activation='relu'))
annModel.add(tf.keras.layers.Dense(units=256, activation='relu', 
                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=regularizers.l2(1e-4),
                                   activity_regularizer=regularizers.l2(1e-5)))
annModel.add(tf.keras.layers.LeakyReLU(alpha=0.05))
annModel.add(tf.keras.layers.Dropout(rate=0.3))
annModel.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
"""
annModel.add(tf.keras.layers.Dense(units=256, activation='relu', 
                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=regularizers.l2(1e-4),
                                   activity_regularizer=regularizers.l2(1e-5)))
"""
annModel.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

annModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

annModel.fit(X_train, y_train, batch_size=16, epochs=200)

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
