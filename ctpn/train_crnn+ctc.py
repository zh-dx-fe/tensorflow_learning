from data import *
from crnn_model import crnn
import shutil
import os
import numpy as np
from PIL import Image

def change_shape_to_32(img):
    h, w = img.shape[0], img.shape[1]
    scale = 32/h
    img = Image.fromarray(img)
    img = img.resize((int(w*scale),32))
    img = np.array(img)
    img = np.expand_dims(img,0)
    return img

train_lr_init = 1e-3
train_lr_end = 1e-6
trainset = crnn_data()
logdir = "E:/github_zdxf/tensorflow_learning/ctpn/log_ctc"
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = 2 * 100000
total_steps = 30 * 100000
input_tensor = tf.keras.layers.Input([32,None,3])
output_tensor = crnn(input_tensor,128,512)
model = tf.keras.Model(input_tensor, output_tensor)
# model.load_weights('E:\github_zdxf\yolo_qqwweee\keras-yolo3\model_data\yolo.h5',by_name=True,skip_mismatch=True)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)
def train_step(image_data, target, label_length, logit_length):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        total_loss = tf.nn.ctc_loss(target,pred_result,label_length,logit_length,logits_time_major=False)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.16f   total_loss: %4.12f" %(global_steps, optimizer.lr.numpy(),total_loss))
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
            tf.summary.scalar("loss/total_loss", total_loss.numpy()[0], step=global_steps)

        writer.flush()


for epoch in range(30):
    for ll in trainset:
        if ll != 'error!' and ll != None:
            sequence, img = ll
            img = change_shape_to_32(img)
            label_length = sequence.shape[0]
            sequence = np.array(sequence).T
            logit_length = int(img.shape[1]//4)
            train_step(img,sequence,[label_length],[logit_length])
        if int(global_steps.value().numpy()) % 100 == 0:
            model.save_weights("E:/github_zdxf/weights/crnn+ctc_weights/{}_{}.h5".format(epoch,int(global_steps.value().numpy())))
    model.save_weights("E:/github_zdxf/weights/crnn+ctc_weights/{}.h5".format(epoch))



