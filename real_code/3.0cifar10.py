#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:08:02 2021

@author: zhouxinfeng
"""

import tensorflow as tf

data = tf.keras.datasets.cifar10
cifar = data.load_data()
cifar_train,cifar_test = cifar
x_train,y_train = cifar_train
x_test,y_test = cifar_test

x_train,x_test = tf.cast(x_train,dtype=tf.float32)/255.0,tf.cast(x_test,dtype=tf.float32)/255
y_train,y_test = tf.cast(y_train,dtype=tf.int32),tf.cast(y_test,dtype=tf.int32)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, [3,3],padding='same',activation=tf.nn.relu,input_shape=[32,32,3]))
model.add(tf.keras.layers.Conv2D(16, [3,3], padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(32, [3,3], padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(32, [3,3], padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train,y_train,64,5,validation_data=(x_test,y_test))

