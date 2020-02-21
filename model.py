#!/usr/bin/env python

import pandas as pd # Needed to read CSV file
import os # Needed to find the path to the data
import ntpath # Needed to split the path
import matplotlib.pyplot as plt # Needed to plot data
import numpy as np # Needed for histogram and other stuff
import random # Needed to shuffle the data array
import csv # Needed to write to a csv file

import tensorflow as tf
import pandas as pd
from skimage import io
#tf.python.control_flow_ops = tf
from keras import models, optimizers, backend
from keras.layers import core, Convolution2D, pooling, Cropping2D,MaxPool2D,Flatten,Dense,BatchNormalization,Activation

#import keras
from sklearn.model_selection import train_test_split
#import pandas as pd


#diviide data into three buckets
#image size of n*m origionally
input_img_size_n = 320
input_img_size_m = 160
cut_top_img = 60    #number of pixels to be cropped from the top
#image size of n*m after croping
crop_img_size_n =  input_img_size_n
crop_img_size_m =   input_img_size_m-cut_top_img

kernel1 = 32       # number of filters for 1st conv layer
kernel1_size = 3    #filter size for 1st conv layer
MaxPool_1 = 2       #max pool size for 1st conv layer

kernel2 = 64        # number of filters for 2nd conv layer
kernel2_size = 3    #filter size for 2nd conv layer
MaxPool_2 = 2       #max pool size for 2nd conv layer

kernel3 = 64         # number of filters for 2nd conv layer
kernel3_size = 3    #filter size for 2nd conv layer
MaxPool_3 = 2       #max pool size for 2nd conv layer

neurons_1   = 500     # number of neurons in the first dense layer
neurons_2   = 100     # number of neurons in the second dense layer
neurons_3 =  50
neurons_out =1  #output neurons
batchsize = 200
epoch = 25
data_dir = 'udacityData'
#model
model = models.Sequential()

#regularization ??
#cutting 100 pixels from the top
model.add(Cropping2D(cropping=((cut_top_img, 0), (0, 0)),input_shape=(input_img_size_m, input_img_size_n, 3)))
#first convolution layer
#model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
#model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))


model.add(Convolution2D(kernel1,kernel1_size,kernel1_size, input_shape = (crop_img_size_m,crop_img_size_n,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (MaxPool_1,MaxPool_1)))
model.add(BatchNormalization())

#second convolution layer
model.add(Convolution2D(kernel2,kernel2_size,kernel2_size))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (MaxPool_2,MaxPool_2)))
model.add(BatchNormalization())

#Third convolution layer
model.add(Convolution2D(kernel3,kernel3_size,kernel3_size))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (MaxPool_3,MaxPool_3)))
model.add(BatchNormalization())

#flatten
model.add(Flatten())

#first dense layer
model.add(Dense(neurons_1, activation='relu'))
model.add(core.Dropout(.5))#added drop out
#second dense layer
model.add(Dense(neurons_2, activation='relu'))
#third dense layer
model.add(Dense(neurons_3, activation='relu'))
model.add(core.Dropout(.5))#added drop out
#output layer

model.add(Dense(neurons_out))


#compile
model.compile(optimizers.Adam(lr=0.001),
              loss='mean_squared_error',
              metrics=['accuracy'])


#extract im and str from csv and test should be 20 %
fd = pd.read_csv(os.path.join(data_dir, 'balanced_data.csv'))
images_train, images_valid, steering_train, steering_valid = train_test_split(fd['image'], fd['steering'], test_size=0.2)
images_train_list = []
images_valid_list = []
steering_train_list = []
steering_valid_list = []
for i in images_train:
  path = os.path.join(data_dir,i)
  img = io.imread(path)
  images_train_list.append(img)
for i in images_valid:
  path = os.path.join(data_dir,i)
  img = io.imread(path)
  images_valid_list.append(img)
 

def initlist(n):
    inner_steering_list = [0] * n
    return inner_steering_list

def steer(steering):
  num_bins = 50
  outter_steering_list = []
  histogram, bins = np.histogram(fd['steering'], num_bins)

  for i in range( len(steering)):
    inner_steering_list = initlist(50)
    cur_index_inner = 0
    for j in range(num_bins):
      if (steering[i] >= bins[j]) and (steering[i] <= bins[j + 1]):
        inner_steering_list[cur_index_inner] = 1
        break
      cur_index_inner += 1
        #else:
          #inner_steering_list.append(0)
    outter_steering_list.append(np.asarray(inner_steering_list))
  return np.asarray(outter_steering_list)

val_steering = steer(steering_valid.values)
train_steering = steer(steering_train.values)

#model.fit(np.asarray(images_train_list), train_steering,batch_size=batchsize,   epochs=epoch, validation_data=(np.asarray(images_valid_list), val_steering),shuffle=True)
model.fit(np.asarray(images_train_list),steering_train.values ,batch_size=batchsize,   epochs=epoch, validation_data=(np.asarray(images_valid_list),steering_valid.values ),shuffle=True)

# SAve the trained model for us to test it on the track
model.save('model.h5')
