# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

dataGenTrain = ImageDataGenerator(rescale = 1./255, shear_range= 0.2, zoom_range=0.2, horizontal_flip=True)
trainingSet = dataGenTrain.flow_from_directory('dataset/train_set', target_size = (64, 64), batch_size = 16, class_mode = 'binary')

dataGenTest = ImageDataGenerator(rescale = 1./255)
testingSet = dataGenTest.flow_from_directory('dataset/test_set', target_size = (64, 64), batch_size = 16, class_mode = 'binary')

cnnModel = tf.keras.models.Sequential()
cnnModel.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu', input_shape=[64, 64, 3]))
cnnModel.add(tf.keras.layers.MaxPool2D(pool_size =2, strides=2))
cnnModel.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu', input_shape=[64, 64, 3]))
cnnModel.add(tf.keras.layers.MaxPool2D(pool_size =2, strides=2))

cnnModel.add(tf.keras.layers.Flatten())
cnnModel.add(tf.keras.layers.Dense(units = 128, activation='relu'))
cnnModel.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

cnnModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnnModel.fit(x = trainingSet, validation_data = testingSet, epochs = 30)

testImage = image.load_img('dataset/helperPhotos/hotdogMaybe1.png', target_size = (64, 64))
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis = 0)
test2Image = image.load_img('dataset/helperPhotos/hotdogMaybe2.jpg', target_size = (64, 64))
test2Image = image.img_to_array(test2Image)
test2Image = np.expand_dims(test2Image, axis = 0)

result = cnnModel.predict(testImage)
result2 = cnnModel.predict(test2Image)
trainingSet.class_indices
if result[0][0] == 0:
  print('hotdog for image 1')
else:
  print('not hotdog for image 1')
if result2[0][0] == 0:
  print('hotdog for image 2')
else:
  print('not hotdog for image 2')
