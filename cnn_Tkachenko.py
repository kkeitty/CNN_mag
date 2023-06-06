# python3 /home/shuvaev/Dropbox/STUDENTS/temp/cnn_Tkachenko.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
#from keras import optimizers

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2
import math




#'TF_ENABLE_ONEDNN_OPTS=0'

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print()
    return np.array(data, dtype=object)

train = get_training_data('/home/shuvaev/Downloads/chest_xray/train')
test = get_training_data('/home/shuvaev/Downloads/chest_xray/test')
val = get_training_data('/home/shuvaev/Downloads/chest_xray/val')

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label+1)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label+1)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label+1)


# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255


# resize data for deep learning
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)



# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


"""def my_loss_fn_1(y_pred, y_true, epsilon=1e-12): # mse
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    predictions = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    ce = -np.sum(y_true * np.log(y_pred + 1e-12))/N
    return ce
"""

"""def my_loss_fn(y_pred, y_true, n=5216): # binomial distribution
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = (y_true / y_pred) - ((n - y_true) / (1 - y_pred))
    return ce
"""

"""def my_loss_fn(y_pred, y_true): # negative binomial distribution !!!!!
    y_true = tf.math.real(tf.cast(y_true, tf.float32))
    y_pred = tf.math.real(tf.cast(y_pred, tf.float32))
    ce = 1 * (1 / (1 - y_pred)) + (y_true / y_pred)
    return tf.math.real(ce)
"""

"""def my_loss_fn(y_pred, y_true): # negative binomial distribution !!!!! + log
    # k = 1
    y_true = tf.math.real(tf.cast(y_true, tf.float32))
    y_pred = tf.math.real(tf.cast(y_pred, tf.float32))
    ce = 1 * log(1 - y_pred) + y_true * log(y_pred)
    return tf.math.real(ce)
"""

"""def my_loss_fn(y_pred, y_true):
    n, p = tf.unstack(y_pred, num=2, axis=-1)
	# Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)
    
    # Calculate the negative log likelihood
    nll = (
        tf.math.lgamma(n) 
        + tf.math.lgamma(y_true + 1)
        - tf.math.lgamma(n + y_true)
        - n * tf.math.log(p)
        - y_true * tf.math.log(1 - p)
    )                  

    return nll
# https://gist.github.com/sfblake/eade3b56e509da5bcc081ab37c4ee69f
"""
"""
def nbinom_pmf_tf(x,n,p):
    coeff = tf.lgamma(n + x) - tf.lgamma(x + 1) - tf.lgamma(n)
    return tf.cast(tf.exp(coeff + n * tf.log(p) + x * tf.log(1 - p)),dtype=tf.float64)

def def my_loss_fn(y_pred, y_true):
    result = tf.map_fn(lambda x: -nbinom_pmf_tf(x[1]
                                                , x[0][0]
                                                , tf.minimum(tf.constant(0.99,dtype=tf.float64),x[0][1]))
                       ,(y_pred,y_true)
                       ,dtype=tf.float64)
    result = tf.reduce_sum(result,axis=0)
    return result
# https://stackoverflow.com/questions/55782674/how-should-i-write-the-loss-function-with-keras-and-tensorflow-for-negative-bi
"""
"""def my_loss_fn(y_pred, y_true): # multinomial distribution
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = sum(y_true / y_pred)
    return ce


def my_loss_fn(y_pred, y_true): # Gamma distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = 2 * (-tf.math.log(y_pred/y_true) + ((y_pred - y_true) / y_pred))
    return ce


# MULTINOMIAL DISTRIBUTION

# by: https://blog.jakuba.net/maximum-likelihood-for-multinomial-distribution/
# logL = log(n)! + sum( x_i log(p_i) ) - sum( log(x_i!) )
# p_i = probability of i-th class, x_i = number of i-th class occurence in the dataset

# by: https://hal.archives-ouvertes.fr/hal-03008622/document
# logL = sum_{k=1^K} sum_{j=ak^(ak+xk-1)} log(j) - sum_{k=A^(A+N-1)} log(i)
# xk = observations, N = sum xk, k = categories, sum k = K, a = positive parameter, sum_k ak = A
# a conjugated with p as fDir(p;a) ~ prod_{k=1^K} p_k^(ai-1)

@tf.function 
def my_loss_fn(y_pred, y_true): # poisson deviance
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	ce = 2 * ( y_true * tf.math.log(y_true/y_pred) - y_true + y_pred )
	return ce
"""


model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units = len(labels), activation='softmax'))

"""
def my_loss_fn(y_pred, y_true): # Gamma distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = 2 * (-tf.math.log(y_pred/y_true) + ((y_pred - y_true) / y_pred))
    return tf.reduce_mean(ce, axis=-1)
"""
"""
def my_loss_fn(y_pred, y_true): # Poisson distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = 2 * (y_pred * tf.math.log(y_pred / y_true) - y_pred + y_true)
    return tf.reduce_mean(ce, axis=-1)
"""
"""
def my_loss_fn(y_pred, y_true): # Binomial distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = 2 * (((y_pred - y_true) ** 2) / y_pred)
    return ce
"""


def my_loss_fn2(y_pred, y_true):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


from tensorflow.keras import backend as K
def correlation_loss(y_true, y_pred):
    x = tf.cast(y_true, tf.float32)
    y = tf.cast(y_pred, tf.float32)
    mx = K.mean(tf.convert_to_tensor(x),axis=1)
    my = K.mean(tf.convert_to_tensor(y),axis=1)
    mx = tf.tile(tf.expand_dims(mx,axis=1),(1,x.shape[1]))
    my = tf.tile(tf.expand_dims(my,axis=1),(1,x.shape[1]))
    xm, ym = (x-mx)/100, (y-my)/100
    r_num = K.sum(tf.multiply(xm,ym),axis=1)
    r_den = tf.sqrt(tf.multiply(K.sum(K.square(xm),axis=1), K.sum(K.square(ym),axis=1)))
    r = tf.reduce_mean(r_num / r_den)
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - r    
def root_mean_squared_error(y_true, y_pred):
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	return K.sqrt(K.mean(K.square(y_pred - y_true)))

def Poisson_loss(y_pred, y_true): # Poisson distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = 2 * (y_pred * tf.math.log(y_pred / y_true) - y_pred + y_true)
    return ce
def aver_Poisson_loss(y_pred, y_true): # Poisson distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = (y_pred * tf.math.log(y_pred / y_true) - y_pred + y_true)
    return 2*tf.reduce_mean(ce)

def Gamma_loss(y_pred, y_true): # Gamma distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = tf.math.log(y_pred / y_true) + (y_pred - y_true) / y_pred
    return 2*ce
def aver_Gamma_loss(y_pred, y_true): # Gamma distribution Deviance
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = tf.math.log(y_pred / y_true) + (y_pred - y_true) / y_pred
    return 2*tf.reduce_mean(ce)
    
model.compile(optimizer = 'adam', loss = Gamma_loss, metrics=['accuracy'])
#model.compile(optimizer = keras.optimizers.Adam(clipvalue=0.5), loss = 'mse', metrics=['accuracy'])

# model.summary()


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
                                            min_lr=0.000001)
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=5,
                    validation_data=datagen.flow(x_val, y_val))#,                    callbacks=learning_rate_reduction)
model.evaluate(x_test, y_test)

