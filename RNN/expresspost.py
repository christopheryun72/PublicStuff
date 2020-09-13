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
import warnings
warnings.filterwarnings("ignore")

dataframe = pd.read_csv('EXPR.csv')
print(dataframe)
dataframe.Date = pd.to_datetime(dataframe.Date)
print(dataframe.Date)
dataframe = dataframe.set_index("Date")

train_data = dataframe[:-30]
test_data = dataframe[-30:]
print(train_data)
print(test_data)

train_data = train_data.iloc[:, 3:4]
test_data = test_data.iloc[:, 3:4]
print(train_data)
print(test_data)

scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
print(train_data)
print(test_data)

test_size = 30
features = 1
generator = TimeseriesGenerator(train_data, train_data, length=test_size, batch_size=32)

rModel = Sequential()
rModel.add(LSTM(units=256, return_sequences = True, input_shape= (test_size, features)))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=256, return_sequences = True))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=256, return_sequences = True))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=256, return_sequences = False))
rModel.add(Dropout(rate=0.15))
rModel.add(Dense(units=1))

rModel.compile(optimizer='adam', loss='mean_squared_error')
rModel.fit_generator(generator, epochs=150)

#Take a Training Batch (3D Np Array of scaled values), predicting off of that batch, and appending that predicted
#value to the end of this batch, while removing the first element from this batch (first elem has relation with pred value)
pred_list = []
batch = train_data[-test_size:].reshape((1, test_size, features))
print(batch)

for i in range(test_size):
  print(rModel.predict(batch)[0])
  print(i)
  pred_list.append(rModel.predict(batch)[0])
  print(pred_list) 
  batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
  print(batch)
  print(i)
pred_list = scaler.inverse_transform(pred_list)

#Creating new dates after the total dataset
from pandas.tseries.offsets import DateOffset
print(dataframe.index[-1])
add_dates = [dataframe.index[-1] + DateOffset(days=x) for x in range(0,31) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=dataframe.columns)

#Creating new pandas dataframe with results in prediction column, indexed by newly produced dates
dataframe_pred = pd.DataFrame(pred_list, index=future_dates[-test_size:].index, columns=['Prediction'])

#Adding on to the original
newDataFrame = pd.concat([dataframe,dataframe_pred], axis=1)
print(newDataFrame)

plt.figure(figsize=(20, 5))
plt.plot(newDataFrame.index, newDataFrame['Close'])
plt.plot(newDataFrame.index, newDataFrame['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
