# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset_train = pd.read_csv('EXPR_train.csv')
training_set = dataset_train.iloc[:, 4:5].values
training_set

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled

X_train = []
y_train = []
for i in range(100, 1198):
  X_train.append(training_set_scaled[i - 100:i, 0])
  y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)
print(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train

rModel = Sequential()
rModel.add(LSTM(units=100, return_sequences=True, input_shape= (X_train.shape[1], 1)))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=100, return_sequences=True))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=100, return_sequences=True))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=100, return_sequences=False))
rModel.add(Dropout(rate=0.15))
rModel.add(Dense(units=1))

rModel.compile(optimizer='adam', loss='mean_squared_error')
rModel.fit(X_train, y_train, batch_size=32, epochs=150)

dataset_test = pd.read_csv('EXPR_test.csv')
real_stock_price = dataset_test.iloc[:, 4:5].values
print(real_stock_price)
len(real_stock_price)

dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
dataset_total = dataset_total[np.logical_not(np.isnan(dataset_total))]
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 100:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
print(inputs)

X_test = []
for i in range(100, 161):
  X_test.append(inputs[i-100:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_stocks_results = rModel.predict(X_test)
pred_stocks_results = sc.inverse_transform(pred_stocks_results)
pred_stocks_results

plt.plot(real_stock_price, color = 'red', label = 'Real EXPR Stock Price')
plt.plot(pred_stocks_results, color = 'blue', label = 'Predicted EXPR Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
