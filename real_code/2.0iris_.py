# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:44:54 2021

@author: zxf
"""
import tensorflow as tf
import numpy as np
import pandas as pd
data_test = pd.read_csv('C:/Users/lenovo/Desktop/_TensorFlow_tf/real_code/data_/iris_test.csv')
data_train = pd.read_csv('C:/Users/lenovo/Desktop/_TensorFlow_tf/real_code/data_/iris_training.csv')
data_test,data_train = np.array(data_test),np.array(data_train)
x_train,x_test = tf.cast(data_train[:,0:4],dtype=tf.float32),tf.cast(data_test[:,:4],dtype=tf.float32)
# print(x_train.shape,x_test.shape)
# x_mean.shape
# x_train.shape
x_mean = tf.reduce_mean(x_train,axis=0)
# x_mean.shape
x_train,x_test = x_train-x_mean , x_test-x_mean
# x_train.shape
y_train,y_test = tf.one_hot(tf.cast(data_train[:,4],dtype=tf.int32),3),tf.one_hot(tf.cast(data_test[:,4],dtype=tf.int32),3)
# y_train.shape
# y_test.shape
np.random.seed(112)
learn_rate = 0.5
iter = 50
display_step = 10

w1 = tf.Variable(np.random.randn(4,16),dtype=tf.float32)
b1 = tf.zeros(shape=[16],dtype=tf.float32)
b1 = tf.Variable(b1)
w2 = tf.Variable(np.random.randn(16,3),dtype=tf.float32)
b2 = tf.Variable(np.zeros(3),dtype=tf.float32)
# w2.shape,b2.shape


acc_train = []
acc_test = []
cce_train = []
cce_test = []



for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        hidden_train = tf.nn.relu(tf.matmul(x_train,w1)+b1)
        pre_train = tf.nn.softmax(tf.matmul(hidden_train,w2)+b2)
        loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=pre_train,y_true=y_train),axis=0)
        hidden_test = tf.nn.relu(tf.matmul(x_test,w1)+b1)
        pre_test = tf.nn.softmax(tf.matmul(hidden_test,w2)+b2)
        loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test,pre_test),axis=0) 
        
    train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre_train.numpy(),axis=1),tf.argmax(y_train.numpy(),axis=1)),dtype=tf.float32))
    test_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre_test.numpy(),axis=1),tf.argmax(y_test.numpy(),axis=1)),dtype=tf.float32))
    
    acc_train.append(train_acc)
    acc_test.append(test_acc)
    cce_train.append(loss_train)
    cce_test.append(loss_test)
    
    grads = tape.gradient(loss_train,[w1,b1,w2,b2])
    w1.assign_sub(learn_rate*grads[0])
    b1.assign_sub(learn_rate*grads[1])
    w2.assign_sub(learn_rate*grads[2])
    b2.assign_sub(learn_rate*grads[3])
    if i % display_step == 0:
        print('i:{},acc_train:{},cce_train:{},acc_test:{},cce_test{}'.format(i,train_acc,loss_train,test_acc,loss_test))

