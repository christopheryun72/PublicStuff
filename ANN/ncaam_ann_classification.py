# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('ncaam.csv')
X = data.loc[:, list(data.columns[1:-3])].values
y = data.iloc[:, -3].values
tempY = []
for val in y:
  if val == "Champions" or val == "2ND" or val == "F4":
    tempY.append(1)
  else:
    tempY.append(0)
y = tempY
y = np.asarray(y)

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sScalar = StandardScaler()
X_train = sScalar.fit_transform(X_train)
X_test = sScalar.transform(X_test)
print(X)
print(y)

annModel = tf.keras.models.Sequential()
annModel.add(tf.keras.layers.Dense(units=6, activation='relu'))
annModel.add(tf.keras.layers.Dense(units=6, activation='relu'))
annModel.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
annModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
annModel.fit(X_train, y_train, batch_size=32, epochs=50)

y_pred = annModel.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Testing if Under or Overfitting (Results)
y_exp = annModel.predict(X_train)
y_exp = (y_exp > 0.5)
print(np.concatenate((y_exp.reshape(len(y_exp),1), y_train.reshape(len(y_train),1)),1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#Testing if Under or Overfitting (%s)
confMatrix2 = confusion_matrix(y_train, y_exp)
accur2 = accuracy_score(y_train, y_exp)
print(confMatrix2)
print(accur2)
