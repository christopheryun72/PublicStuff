# -*- coding: utf-8 -*-


import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

dataset = pd.read_csv('debateSpeaker.csv', encoding = "ISO-8859-1", engine='python')
dataset = dataset.sample(frac=1).reset_index(drop=True)
print("Shape of Dataset:", dataset.shape)
dataset.head()

dataset = dataset[['Text', 'Speaker']]
dataset.head()

dataset['Speaker'].value_counts().sort_index().plot.bar()

dataset['Text'].str.len().plot.hist()

dataset['Text'] = dataset['Text'].apply(lambda x: x.lower())
dataset.head()

tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(dataset['Text'].values)
X = tokenizer.texts_to_sequences(dataset['Text'].values)
X = sequence.pad_sequences(X)
X[:5]

msaModel = tf.keras.models.Sequential()
msaModel.add(tf.keras.layers.Embedding(5000, 128, input_length= X.shape[1]))
msaModel.add(tf.keras.layers.Dropout(rate=0.2))
msaModel.add(tf.keras.layers.LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
msaModel.add(tf.keras.layers.LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
msaModel.add(tf.keras.layers.LSTM(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
msaModel.add(tf.keras.layers.Dense(12, activation='softmax'))
msaModel.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
msaModel.summary()

y = pd.get_dummies(dataset['Speaker']).values
[print(dataset['Speaker'][i], y[i]) for i in range(5)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
msaModel.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

msaModel.save('presidential_debate_speaker.h5')

predictions = msaModel.predict(X_test)
[print(dataset['Text'][i], predictions[i], y_test[i]) for i in range(5)]

"""
Although there are twelve different speakers, 
deciding to only analyze the accuracy of the more important speakers.
"""
pred_clinton, pred_cooper, pred_holt, pred_kaine, pred_pence, pred_trump, pred_wallace = 0,0,0,0,0,0,0
real_clinton, real_cooper, real_holt, real_kaine, real_pence, real_trump, real_wallace = 0,0,0,0,0,0,0
for i, prediction in enumerate(predictions):
  if np.argmax(prediction) == 2:
    pred_clinton += 1
  elif np.argmax(prediction) == 3:
    pred_cooper += 1
  elif np.argmax(prediction) == 4:
    pred_holt += 1
  elif np.argmax(prediction) == 5:
    pred_kaine += 1
  elif np.argmax(prediction) == 6:
    pred_pence += 1
  elif np.argmax(prediction) == 10:
    pred_trump += 1
  elif np.argmax(prediction) == 11:
    pred_wallace += 1
  
  if np.argmax(y_test[i]) == 2:
    real_clinton += 1
  elif np.argmax(y_test[i]) == 3:
    real_cooper += 1
  elif np.argmax(y_test[i]) == 4:
    real_holt += 1
  elif np.argmax(y_test[i]) == 5:
    real_kaine += 1
  elif np.argmax(y_test[i]) == 6:
    real_pence += 1
  elif np.argmax(y_test[i]) == 10:
    real_trump += 1
  elif np.argmax(y_test[i]) == 11:
    real_wallace += 1

print('Clinton Predictions:', pred_clinton)
print('Clinton Actual:', real_clinton)

print('Cooper Predictions:', pred_cooper)
print('Cooper Actual:', real_cooper)

print('Holt Predictions:', pred_holt)
print('Holt Actual:', real_holt)

print('Kaine Predictions:', pred_kaine)
print('Kaine Actual:', real_kaine)

print('Pence Predictions:', pred_pence)
print('Pence Actual:', real_pence)

print('Trump Predictions:', pred_trump)
print('Trump Actual:', real_trump)

print('Wallace Predictions:', pred_wallace)
print('Wallace Actual:', real_wallace)
