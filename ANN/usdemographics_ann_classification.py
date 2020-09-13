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
print(testing)
type(testing)

data = pd.read_csv('data_US.csv')
X = data.loc[:, list(data.columns[3:11]) + list(data.columns[12:])].values
y = data.iloc[:, 11].values
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = keras.utils.to_categorical(encoded_Y)
print(y)
print(y[0][30:45])

le = LabelEncoder()
for i in range(2, 55):
  X[:, i] = le.fit_transform(X[:, i])

print(X)
ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct2.fit_transform(X.astype(str)))
print(X)

X = X.astype(np.float)
print(X[6, :])
y = y.astype(np.float)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train[6, :20])
print(y)

annModel = tf.keras.models.Sequential()
#annModel.add(tf.keras.layers.Dense(units = 256, activation='relu', input_dim = 56))
annModel.add(tf.keras.layers.Dense(units=256, activation='relu', 
                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=regularizers.l2(1e-4),
                                   activity_regularizer=regularizers.l2(1e-5)))
annModel.add(tf.keras.layers.Dropout(rate=0.5))
#annModel.add(tf.keras.layers.Dense(units = 256, activation='relu'))
annModel.add(tf.keras.layers.Dense(units=256, activation='relu', 
                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=regularizers.l2(1e-4),
                                   activity_regularizer=regularizers.l2(1e-5)))
annModel.add(tf.keras.layers.Dropout(rate=0.5))
annModel.add(tf.keras.layers.Dense(units = 48, activation='softmax'))
print(y[:10])

X_train[50:51]

print(len(y_train))
len(y_train[50:51][0])

annModel.compile(optimizer='adam', loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

annModel.fit(X_train, y_train, batch_size=8, epochs=100)

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

#If using StratifiedKFold Cross-Validation, need to use this helper method
def getNewLabels(y):
  newY = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
  return newY
print(y)
yLabeled = getNewLabels(y)
print(yLabeled)

#Confirming Overfitting via Stratified K-Fold Cross-Validation
folds = StratifiedKFold(n_splits = 10)
scores = []
for train_index, test_index in folds.split(X, yLabeled):
  X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #If wanting to test on multiple, different models, line below but with diff model name and obviously a different container
  scores.append(get_score(annModel, X_train, X_test, y_train, y_test))
scores
