# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('covidClorox.csv')
print(dataset)
dataset.Date = pd.to_datetime(dataset.Date)
dataset = dataset.set_index("Date")
print(dataset)

train_data = dataset[:-15]
test_data = dataset[-15:]
print(train_data)
print(test_data)

scalar = MinMaxScaler()
scalar.fit(train_data)
train_data = scalar.transform(train_data)
test_data = scalar.transform(test_data)
print(train_data)
print(test_data)

test_size = 15
features = 2
generator = TimeseriesGenerator(train_data, train_data, length=test_size, batch_size=8)

rModel = Sequential()
rModel.add(LSTM(units=512, return_sequences = True, input_shape= (test_size, features), recurrent_dropout=0.1, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
rModel.add(Dropout(rate=0.20))
rModel.add(LSTM(units=512, return_sequences = True, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
rModel.add(Dropout(rate=0.20))
rModel.add(LSTM(units=512, return_sequences = False, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
rModel.add(Dropout(rate=0.20))
rModel.add(Dense(units=1))

rModel.compile(optimizer='RMSProp', loss="mean_squared_error")
rModel.fit_generator(generator, epochs=100)

pred_list = []
batch = train_data[-test_size:].reshape((1, test_size, features))
for i in range(test_size):
  pred_list.append(rModel.predict(batch)[0])
  batch = batch[:, 1:, :]
print(pred_list)
newScalar = MinMaxScaler()
newScalar.min_, newScalar.scale_ = scalar.min_[0], scalar.scale_[0]
pred_list = newScalar.inverse_transform(pred_list)
print(pred_list)

from pandas.tseries.offsets import DateOffset
add_dates = [dataset.index[-1] + DateOffset(days=x) for x in range(0, 16)]
future_dates = pd.DataFrame(index=add_dates[1:], columns=dataset.columns)

dataframe_pred = pd.DataFrame(pred_list, index=future_dates[-test_size:].index, columns=['Prediction'])
newDataFrame = pd.concat([dataset,dataframe_pred], axis=1)
print(newDataFrame)

plt.figure(figsize=(20, 5))
plt.plot(newDataFrame.index, newDataFrame['CovidNew'], color='r')
plt.plot(newDataFrame.index, newDataFrame['CloroxStock'], color='b')
plt.plot(newDataFrame.index, newDataFrame['Prediction'], color='g')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
