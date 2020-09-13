# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_excel('LD.xlsx')
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values
#print(X)
#print(y)

labelE = LabelEncoder()
X[:, 1] = labelE.fit_transform(X[:, 1])
#print(X)
labelE2 = LabelEncoder()
X[:, 0] = labelE2.fit_transform(X[:, 0])
#print(X)

cTransformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [2])], remainder = 'passthrough')
X = np.array(cTransformer.fit_transform(X))
#print(X)

cTransformer2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [5])], remainder = 'passthrough')
X = np.array(cTransformer2.fit_transform(X))
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

sScaler = StandardScaler()
X_train = sScaler.fit_transform(X_train)
X_test = sScaler.transform(X_test)

annModel = tf.keras.models.Sequential()
annModel.add(tf.keras.layers.Dense(units = 12, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 12, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
annModel.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

annModel.fit(X_train, y_train, batch_size=2, epochs = 75)

y_pred = annModel.predict(X_test)
y_pred = (y_pred > 0.5)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Testing if Under or Overfitting (Results)

y_exp = annModel.predict(X_train)
y_exp = (y_exp > 0.5)
#print(np.concatenate((y_exp.reshape(len(y_exp),1), y_train.reshape(len(y_train),1)),1))

confMatrix = confusion_matrix(y_test, y_pred)
accur = accuracy_score(y_test, y_pred)
#print(confMatrix)
#print(accur)

#Testing if Under or Overfitting (%s)
confMatrix2 = confusion_matrix(y_train, y_exp)
accur2 = accuracy_score(y_train, y_exp)
#print(confMatrix2)
#print(accur2)
