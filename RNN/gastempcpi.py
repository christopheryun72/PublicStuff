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
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

dataframe = pd.read_csv('GasTempCPI.csv')
print(dataframe)
dataframe.Date = pd.to_datetime(dataframe.Date, dayfirst=True)
dataframe = dataframe.set_index("Date")
dataframe = dataframe[:7750]
print(dataframe)

print(dataframe['Temperature'].isnull().values.any())
print(dataframe['Fuel_Price'].isnull().values.any())
print(dataframe['CPI'].isnull().values.any())
print(dataframe['Unemployment'].isnull().values.any())

imrCPI = SimpleImputer(missing_values=np.nan, strategy='mean')
imrCPI = imrCPI.fit(dataframe[['CPI']])
dataframe['CPI'] = imrCPI.transform(dataframe[['CPI']])
print(dataframe['CPI'].isnull().values.any())

imrUE = SimpleImputer(missing_values=np.nan, strategy='mean')
imrUE = imrUE.fit(dataframe[['Unemployment']])
dataframe['Unemployment'] = imrUE.transform(dataframe[['Unemployment']])
print(dataframe['Unemployment'].isnull().values.any())
print(dataframe)

temp = dataframe.iloc[:, 1:2].values
temp = np.reshape(temp, len(temp))
print(temp)

fuelPrice = dataframe.iloc[:, 2:3].values
fuelPrice = np.reshape(fuelPrice, len(fuelPrice))
print(fuelPrice)
cpi = dataframe.iloc[:, -3:-2].values
cpi = np.reshape(cpi, len(cpi))
print(cpi)
unemployment = dataframe.iloc[:, -2:-1].values
unemployment = np.reshape(unemployment, len(unemployment))
print(unemployment)

test1 = np.corrcoef(temp, fuelPrice)
test2 = np.corrcoef(temp, cpi)
test3 = np.corrcoef(temp, unemployment)
print(test1)
print(test2)
print(test3)

#Choosing Temp and FuelPrice as Indicators to show how little/no correlation skews data
dataframe = dataframe.iloc[:, 1:3]
train_data = dataframe.iloc[:-100]
test_data = dataframe.iloc[-100:]
print(train_data)
print(test_data)

scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
print(train_data)
print(test_data)

test_size = 100
features = 2
generator = TimeseriesGenerator(train_data, train_data, length=test_size, batch_size=64)

rModel = Sequential()
rModel.add(LSTM(units=256, return_sequences = True, input_shape= (test_size, features)))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=256, return_sequences = True))
rModel.add(Dropout(rate=0.15))
rModel.add(LSTM(units=256, return_sequences = False))
rModel.add(Dropout(rate=0.15))
rModel.add(Dense(units=1))

rModel.compile(optimizer='RMSProp', loss='mean_squared_error')
rModel.fit_generator(generator, epochs=50)

#Take a Training Batch (3D Np Array of scaled values), predicting off of that batch, and appending that predicted
#value to the end of this batch, while removing the first element from this batch (first elem has relation with pred value)
pred_list = []
batch = train_data[-test_size:].reshape((1, test_size, features))
#print(batch)

for i in range(test_size):
  pred_list.append(rModel.predict(batch)[0])
  batch = batch[:, 1:, :]
  #batch = np.append(batch[:,1:,:], )
print(pred_list)
newScale = MinMaxScaler()
newScale.min_, newScale.scale_=scaler.min_[0], scaler.scale_[0]
pred_list = newScale.inverse_transform(pred_list)
print(pred_list)

#Creating new dates after the total dataset
from pandas.tseries.offsets import DateOffset
print(dataframe.index[-1])
add_dates = [dataframe.index[-1] + DateOffset(weeks=x) for x in range(0,101)]
future_dates = pd.DataFrame(index=add_dates[1:],columns=dataframe.columns)
print(pred_list)
print(len(pred_list))

#Creating new pandas dataframe with results in prediction column, indexed by newly produced dates
dataframe_pred = pd.DataFrame(pred_list, index=future_dates[-test_size:].index, columns=['Prediction'])
print(dataframe)
print(dataframe_pred)
#Adding on to the original
print(dataframe.shape)
print(dataframe_pred.shape)
#Below threw off graph, shouldn't have concatenated this way
#Added to the right of the original, but not offset to the bottom set
dataframe.reset_index(drop=True, inplace=True)
dataframe_pred.reset_index(drop=True, inplace=True)
newDataFrame = pd.concat([dataframe,dataframe_pred], axis=1)
print(newDataFrame)

plt.figure(figsize=(50, 20))
plt.plot(newDataFrame.index, newDataFrame['Temperature'], color='r')
plt.plot(newDataFrame.index, newDataFrame['Fuel_Price'], color='b')
plt.plot(newDataFrame.index, newDataFrame['Prediction'], color='g')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
"""
Verdict:
As expected, since the Temperature and Gas Prices were not correlated,
the prediction was wonky. The prediction line should be on the right part of the graph,
but messed up concatenation of the final dataframe of the original to the future data.
"""
