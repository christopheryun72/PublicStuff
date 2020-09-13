# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.__version__

data = pd.read_excel('CA.xlsx')
X = data.iloc[:, 2:-1].values
y = data.iloc[:, -1].values
#avg_temp in celsius, p1-p3 are in grams

data.info()
#data.describe()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print(type(y_test))

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 12, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 12, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1))

ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train, y_train, batch_size = 2, epochs = 75)

y_pred = ann.predict(X_test)
#Seeing Accuracy of Model
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Testing if Under or Overfitting
"""
y_exp = ann.predict(X_train)
print(np.concatenate((y_exp.reshape(len(y_exp),1), y_train.reshape(len(y_train),1)),1))
"""
