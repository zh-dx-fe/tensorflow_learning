import tensorflow as tf
from data import img_data
from model import *
import shutil
import os
import numpy as np



train_lr_init = 1e-3
train_lr_end = 1e-6
trainset = img_data('train')
logdir = "E:/github_zdxf/tensorflow_learning/ctpn/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = 2 * steps_per_epoch
total_steps = 30 * steps_per_epoch
input_tensor = tf.keras.layers.Input([408,408,3])
conv_feat = conv_feat_layers(input_tensor,408,True)
output_tensors = rnn_detect_layers(conv_feat,num_anchors=4)
model = tf.keras.Model(input_tensor, output_tensors)
# model.load_weights('E:\github_zdxf\yolo_qqwweee\keras-yolo3\model_data\yolo.h5',by_name=True,skip_mismatch=True)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)
def train_step(image_data, target):#target:((batch_label_sbbox, batch_sbboxes), (batch_label_mbbox, batch_mbboxes), (batch_label_lbbox, batch_lbboxes))
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        total_loss = detect_loss(pred_result[0],pred_result[1],pred_result[2],target[0],target[1],target[2])

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.12f   total_loss: %4.8f" %(global_steps, optimizer.lr.numpy(),total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *train_lr_init
        else:
            lr = train_lr_end + 0.5 * (train_lr_init - train_lr_end) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)

        writer.flush()


for epoch in range(30):
    for ll in trainset:
        if ll != 'error!':
            img, [height_feat, width_feat], target_cls, target_ver, target_hor = ll
            train_step(img, [target_cls,target_ver,target_hor])
        if int(global_steps.value().numpy()) % 100 == 0:
            model.save_weights("E:/github_zdxf/weights/ctpn_weights/{}_{}.h5".format(epoch,int(global_steps.value().numpy())))
    model.save_weights("E:/github_zdxf/weights/ctpn_weights/{}.h5".format(epoch))