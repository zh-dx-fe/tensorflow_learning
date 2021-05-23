import model
import tensorflow as tf

# conv_1 = model.CONV(32,3,2)
# res_1 = model.Res(32)
# model1 = tf.keras.models.Sequential()
# model1.add(tf.keras.layers.Conv2D(32,3,2,padding='same',input_shape=[416,416,3],activation=tf.keras.activations.relu))
# model1.add(conv_1)
# model1.add(res_1)
model1 = model.Darknet()
model1.build(input_shape=[None,416,416,3])
