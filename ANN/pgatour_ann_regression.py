# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('pgaTourData.csv')
imrY = SimpleImputer(missing_values=np.nan, strategy='mean')
imrY = imrY.fit(data[['Money']])
data['Money'] = imrY.transform(data[['Money']])

X = data.loc[:, list(data.columns[1:3]) + list(data.columns[4:16])].values
y = data.iloc[:, -1].values

imr = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0)
imr = imr.fit(X[:, 8:10])
X[:, 8:10] = imr.transform(X[:, 8:10])

imrAnother = SimpleImputer(missing_values=np.nan, strategy='mean')
imrAnother = imrAnother.fit(X[:, 7:8])
X[:, 7:8] = imrAnother.transform(X[:, 7:8])
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)

annModel = tf.keras.models.Sequential()
annModel.add(tf.keras.layers.Dense(units = 128, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 128, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 128, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 128, activation='relu'))
annModel.add(tf.keras.layers.Dense(units = 1))

annModel.compile(optimizer='adam', loss='mean_squared_error')
annModel.fit(X_train, y_train, batch_size=2, epochs=1000)

y_pred = annModel.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
