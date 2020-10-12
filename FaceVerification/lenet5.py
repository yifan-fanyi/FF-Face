# 2020.06.05
# lenet5 model train on image pairs by directly concatenate them
import numpy as np
import csv
import os
import keras
import sys
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from framework.utli import read_dataset, lfw_train_test

server = 1
if server == True:
    root = '/mnt/yifan/face/'
    pair_txt_path ='data/pairs.txt'
    e = 1
else:
    root = '../'
    pair_txt_path = 'data/mini_pairs.txt'
    e = 1

images_path = "data/HEFrontalizedLfw2/"

foldnum = 1
if len(sys.argv) > 2:
    foldnum = int(sys.argv[1])
    print("\n   <Setting> 10-fold validation. Use: fold%s"%(foldnum))
else:
    print("\n   <Warning> 10-fold validation. No fold number provided, use: fold%s as default!"%(foldnum))

print("   <Setting> root location: %s"%(root))
print("   <Setting> relative image location: %s"%(images_path))
print("   <Setting> relative pairs.txt location: %s\n"%(pair_txt_path))

raw_images, flipped_raw_images, raw_labels = read_dataset(path=root+images_path, size=32)
trainData1, trainData2, train_labels, testData1, testData2, test_labels = lfw_train_test(root, pair_txt_path, raw_images, flipped_raw_images, raw_labels, foldnum, includeflipped=True)

train_data = np.concatenate((trainData1, trainData2), axis=-1) / 255.0
test_data = np.concatenate((testData1, testData2), axis=-1) /255.0
train_labels = np_utils.to_categorical(train_labels, 2)
test_labels = np_utils.to_categorical(test_labels, 2)

model = Sequential()
model.add(Convolution2D(filters = 20, kernel_size = (5, 5), padding = "same", input_shape = (32, 32, 2)))
model.add(Activation(activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides =  (2, 2)))
model.add(Convolution2D(filters = 50, kernel_size = (5, 5), padding = "same"))
model.add(Activation(activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation(activation = "relu"))
model.add(Dense(50))
model.add(Activation(activation = "relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.RMSprop(), metrics = ["accuracy"])

history = model.fit(train_data, train_labels, batch_size = 128, nb_epoch = 100, verbose = 1, validation_data=(test_data, test_labels),shuffle=True)

(loss, accuracy) = model.evaluate(test_data, test_labels, batch_size = 128, verbose = 1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy aug 16_40_140_60')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss aug')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
