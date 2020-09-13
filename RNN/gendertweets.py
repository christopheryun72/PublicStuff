# -*- coding: utf-8 -*-


import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

dataset = pd.read_csv('genderClass.csv', encoding = "ISO-8859-1", engine='python')
dataset = dataset.sample(frac=1).reset_index(drop=True)
print("Shape of Dataset:", dataset.shape)
print(dataset.head())

#Minimzing the Dataset down to columns that matter through the column names
dataset = dataset[['text', 'gender']].iloc[:5000, :]
dataset.head()

#Plotting the Number of Each Value in this particular column... sort_index unnecessary,
#just sorting by indices outputted by value_counts
dataset['gender'].value_counts().sort_index().plot.bar()

dataset['text'].str.len().plot.hist()

dataset['text'] = dataset['text'].apply(lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x))
dataset.head()

tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(dataset['text'].values)
X = tokenizer.texts_to_sequences(dataset['text'].values)
X = sequence.pad_sequences(X)
X[:5]

msaModel = tf.keras.models.Sequential()
msaModel.add(tf.keras.layers.Embedding(5000, 256, input_length= X.shape[1]))
msaModel.add(tf.keras.layers.Dropout(rate=0.3))
msaModel.add(tf.keras.layers.LSTM(units=256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
msaModel.add(tf.keras.layers.LSTM(units=256, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))
msaModel.add(tf.keras.layers.Dense(4, activation='softmax'))
msaModel.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
msaModel.summary()

y = pd.get_dummies(dataset['gender']).values
[print(dataset['gender'][i], y[i]) for i in range(5)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
msaModel.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

msaModel.save('genderClass.h5')

predictions = msaModel.predict(X_test)
[print(dataset['text'][i], predictions[i], y_test[i]) for i in range(5)]

#Way to Judge Accuracy
pred_brand, pred_female, pred_male, pred_unknown = 0, 0, 0, 0
real_brand, real_female, real_male, real_unknown = 0, 0, 0, 0
for i, prediction in enumerate(predictions):
  if np.argmax(prediction) == 0:
    pred_brand += 1
  elif np.argmax(prediction) == 1:
    pred_female += 1
  elif np.argmax(prediction) == 2:
    pred_male += 1
  else:
    pred_unknown +=1
  
  if np.argmax(y_test[i]) == 0:
    real_brand += 1
  elif np.argmax(y_test[i]) == 1:
    real_female += 1
  elif np.argmax(y_test[i]) == 2:
    real_male += 1
  else:
    real_unknown +=1

print('Brand Predictions:', pred_brand)
print('Brand Actual:', real_brand)

print('Female Predictions:', pred_female)
print('Female Actual:', real_female)

print('Male Predictions:', pred_male)
print('Male Actual:', real_male)

print('Unknown Predictions:', pred_unknown)
print('Unknown Actual:', real_unknown)
