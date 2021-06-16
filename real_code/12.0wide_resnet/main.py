import logging

#设置日志输出级别
logging.basicConfig(level=logging.INFO)

#引入第三方组件包
import pandas as pd
import sys
import argparse
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.io import loadmat

from wide_resnet import WideResNet
from utils import load_data
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

#本地数据集文件
input_mat_path = "/Users/zhouxinfeng/Desktop/data/person_face_data/dataset/utkface_small_0.1.mat"
#加载数据集
meta = loadmat(input_mat_path)
#性别数据
gender_list = meta["gender"][0]
#年龄数据
age_list = meta["age"][0]
#人脸数据
image_list = meta["image"]

print("data length:",len(image_list))


#批大小
batch_size=1
#训练次数
nb_epochs=1
#学习率
lr=0.01
#学习率优化方法
opt_name='sgd' #or adam
#宽残差网络深度
depth=16 #10, 16, 22, 28
#宽残差网络宽度
k=1
#验证集比例
validation_split=0.1
#是否进行数据增强
use_augmentation=False
#模型输出路径
output_folder="models"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

logging.info("Loading data...")

X_data, y_data_g, y_data_a, _, image_size, _ = load_data(input_mat_path)
# y_data_g = tf.one_hot(y_data_g, 2)
# y_data_a = tf.one_hot(y_data_a, 101)


#生成训练集、测试集数据
data_num = len(X_data)
indexes = np.arange(data_num)
#随机混淆数据
np.random.shuffle(indexes)
# 由于模型复杂，这里只使用20张图片进行训练，供演示用
X_data = X_data[indexes][:20]
y_data_g = y_data_g[indexes][:20]
y_data_a = y_data_a[indexes][:20]
y_data_g = tf.one_hot(y_data_g, 2)
y_data_a = tf.one_hot(y_data_a, 101)
#构建输入
train_num = int(len(X_data) * (1 - validation_split))

X_train = X_data[:train_num]
X_test = X_data[train_num:]
#构建输出
y_train_g = y_data_g[:train_num]
y_test_g = y_data_g[train_num:]
y_train_a = y_data_a[:train_num]
y_test_a = y_data_a[train_num:]

print("train_num:",train_num,"eval_num:",len(X_test))

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008

def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")

#定义WideResNet网络结构
model = WideResNet(image_size, depth=depth, k=k)()
opt = get_optimizer(opt_name, lr)
model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
              metrics=['accuracy'])

logging.info("Model summary...")
model.summary()

#训练过程回调
callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
             ModelCheckpoint(str(output_folder) + "/weights.hdf5",
                             monitor="val_loss",
                             verbose=1,
                             save_best_only=True,
                             mode="auto")
             ]

logging.info("Running training...")

if use_augmentation:
    # 图像增强
    datagen = ImageDataGenerator(
        width_shift_range=0.1,  # 宽偏移
        height_shift_range=0.1,  # 高偏移
        horizontal_flip=True,  # 水平翻转
        preprocessing_function=get_random_eraser(v_l=0, v_h=255))  # 随机移除

    # 混合输入和输出
    training_generator = MixupGenerator(X_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2,
                                        datagen=datagen)()
    hist = model.fit_generator(generator=training_generator,
                               steps_per_epoch=train_num // batch_size,
                               validation_data=(X_test, [y_test_g, y_test_a]),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)
else:
    hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
                     validation_data=(X_test, [y_test_g, y_test_a]))

df = hist.history
plt.plot(df["pred_gender_loss"], label="loss (gender)")
plt.plot(df["pred_age_loss"], label="loss (age)")
plt.plot(df["val_pred_gender_loss"], label="val_loss (gender)")
plt.plot(df["val_pred_age_loss"], label="val_loss (age)")
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(df["pred_gender_accuracy"], label="accuracy (gender)")
plt.plot(df["pred_age_accuracy"], label="accuracy (age)")
plt.plot(df["val_pred_gender_accuracy"], label="val_accuracy (gender)")
plt.plot(df["val_pred_age_accuracy"], label="val_accuracy (age)")
plt.xlabel("number of epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()