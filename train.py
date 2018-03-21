from __future__ import division
from keras.layers import LSTM,Input,Lambda,Dense,Activation,Conv2D,MaxPooling2D,Dropout,Flatten
from keras import backend as K
from keras.models import Model
import keras
from keras import metrics
from keras.datasets import mnist
import numpy as np
import cv2
import os
from tqdm import tqdm
import random

num_classes = 10
batch_size = 128
epochs = 50
path = "/media/sai/New Volume1/Practice/github_codes/cluttered/mnist-cluttered-master/train/"
images = os.listdir("/media/sai/New Volume1/Practice/github_codes/cluttered/mnist-cluttered-master/train/")
random.shuffle(images)
image_train = images[:int(0.8*len(images))]
image_valid = images[int(0.8*len(images)):]
def generator_train(batch_size,valid=False):
	x_train = np.zeros((batch_size,100,100))
	y_train = np.zeros((batch_size,))
	if valid:
		image = image_valid
	else:
		image = image_train
	random.shuffle(image)
	while(True):
		for i,img in enumerate(image):
			x_train[i%batch_size,:,:] = cv2.imread(path+img,0)
			y_train[i%batch_size] = int(img.split(".")[0].split("_")[-1])
			if i%batch_size==0 and i!=0:
				x_train = x_train/255
				y_train = keras.utils.to_categorical(y_train, num_classes)
				x_train = x_train.reshape(x_train.shape[0], 100, 100, 1)
				yield(x_train,y_train)
				x_train = np.zeros((batch_size,100,100))
				y_train = np.zeros((batch_size,))
x_input = Input(shape=(100,100,1))
soft = Conv2D(32, kernel_size=(3, 3),activation='relu',padding="same")(x_input)
soft = Conv2D(64, (3, 3), activation='relu',padding="same")(soft)
soft = Conv2D(1, (3, 3), activation='relu',padding="same")(soft)
x_mul = keras.layers.Multiply()([x_input,soft])
#conv = Conv2D(32, kernel_size=(3, 3),activation='relu')(x_mul)
#conv = Conv2D(64, (3, 3), activation='relu')(conv)
conv = MaxPooling2D(pool_size=(2, 2))(x_mul)
conv = Dropout(0.25)(conv)
conv = Flatten()(conv)
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.5)(conv)
conv = Dense(num_classes, activation='softmax')(conv)

model = Model(x_input,conv)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="rmsprop",
              metrics=['accuracy'])

model.fit_generator(generator_train(128,False),
          steps_per_epoch=int(16000/128),
          epochs = 10,
          verbose=1,validation_data=generator_train(128,True),validation_steps=int(4000/128))

model.save_weights("model.h5")