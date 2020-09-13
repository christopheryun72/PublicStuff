# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

dataGenTrain = ImageDataGenerator(rescale = 1./255, shear_range= 0.2, zoom_range=0.2, horizontal_flip=True)
trainingSet = dataGenTrain.flow_from_directory('dataset/train_set', target_size = (64, 64), batch_size = 16, class_mode = 'categorical')

dataGenTest = ImageDataGenerator(rescale = 1./255)
testingSet = dataGenTest.flow_from_directory('dataset/test_set', target_size = (64, 64), batch_size = 16, class_mode = 'categorical')

cnnModel = tf.keras.models.Sequential()
cnnModel.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2, activation='relu', input_shape=[64, 64, 3]))
cnnModel.add(tf.keras.layers.MaxPool2D(pool_size =2, strides=2))
cnnModel.add(tf.keras.layers.Dropout(rate=0.05))
cnnModel.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2, activation='relu', input_shape=[64, 64, 3]))
cnnModel.add(tf.keras.layers.MaxPool2D(pool_size =2, strides=2))
cnnModel.add(tf.keras.layers.Dropout(rate=0.05))

cnnModel.add(tf.keras.layers.Flatten())
cnnModel.add(tf.keras.layers.Dense(units = 256, activation='relu'))
cnnModel.add(tf.keras.layers.Dense(units = 6, activation = 'softmax'))

cnnModel.compile(optimizer='adam', loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
cnnModel.fit(x = trainingSet, validation_data = testingSet, epochs = 30)

testImage = image.load_img('dataset/helperPhotos/BurgerKing/ankamall_image_30.jpg', target_size = (64, 64))
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis = 0)
test2Image = image.load_img('dataset/helperPhotos/McDonalds/ankamall_image_10.jpg', target_size = (64, 64))
test2Image = image.img_to_array(test2Image)
test2Image = np.expand_dims(test2Image, axis = 0)

result = cnnModel.predict(testImage)
result2 = cnnModel.predict(test2Image)
print(trainingSet.class_indices)
print(result)
print(result2)
if result[0][0] == 1:
    print('Image is BK')
elif result[0][1] == 1:
    print('Image is KFC')
elif result[0][2] == 1:
    print('Image is McDonalds')
elif result[0][3] == 1:
    print('Image is Other')
elif result[0][4] == 1:
    print('Image is Starbucks')
elif result[0][5] == 1:
    print('Image is Subway')
else:
    print('Something is wrong...')
