import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
# a = 1
(train_x,train_y),(test_x,test_y) = mnist.load_data()
x_train,x_test = tf.cast(train_x,dtype=tf.float32)/255.0 , tf.cast(test_x,dtype = tf.float32)/255.0
# x_train.shape
# x_test.shape
y_train,y_test = tf.cast(train_y,dtype=tf.int32),tf.cast(test_y,dtype=tf.int32)
# y_train.shape,y_test.shape

x_train = tf.reshape(tensor=x_train,shape = [60000,28,28,1])
# x_train.shape
x_test = tf.reshape(tensor=x_test,shape=[10000,28,28,1])
# x_test.shape

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16,kernel_size=[3,3],padding='same',activation=tf.nn.relu,input_shape = [28,28,1]))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2,2]))
model.add(tf.keras.layers.Conv2D(32,kernel_size=[3,3],padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2,2]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
# model.summary()
model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=5,validation_split=0.2)