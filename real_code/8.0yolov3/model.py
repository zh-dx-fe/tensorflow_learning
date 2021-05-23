import tensorflow as tf
import numpy as np
import cv2


class CONV(tf.keras.Model):
    def __init__(self,filter,size,stride):
        super(CONV, self).__init__()
        self.filter = filter
        self.size = size
        self.stride = stride
        self.conv = tf.keras.layers.Conv2D(self.filter,self.size,self.stride,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn1(x,training = None)
        x = tf.nn.leaky_relu(x)
        return x

class Res(tf.keras.Model):
    def __init__(self,filter):
        super(Res, self).__init__()
        self.filter = filter
        self.conv1 = CONV(self.filter/2,1,1)
        self.conv2 = CONV(self.filter,3,1)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = x + residual
        return x


class Darknet(tf.keras.Model):
    def __init__(self):
        super(Darknet, self).__init__()
        self.conv1 = CONV(32,3,1)
        self.conv2 = CONV(64,3,2)
        self.res3 = Res(64)
        self.conv4 = CONV(128,3,2)
        # self.res5 = Res(128)
        self.blocks5 = tf.keras.models.Sequential()
        self.conv6 = CONV(256,3,2)
        # self.res7 = Res(256)
        self.blocks7 = tf.keras.models.Sequential()
        self.conv8 = CONV(512,3,2)
        # self.res9 = Res(512)
        self.blocks9 = tf.keras.models.Sequential()
        self.conv10 = CONV(1024,3,2)
        # self.res11 = Res(1024)
        self.blocks11 = tf.keras.models.Sequential()
        for i in range(2):
            self.blocks5.add(Res(128))
        for i in range(8):
            self.blocks7.add(Res(256))
        for i in range(8):
            self.blocks9.add(Res(512))
        for i in range(4):
            self.blocks11.add(Res(1024))

        self.conv12 = CONV(512,1,1)
        self.conv13 = CONV(1024,3,1)
        self.conv14 = CONV(512,1,1)
        self.conv15 = CONV(1024,3,1)
        self.conv16 = CONV(512,1,1)
        self.conv17 = CONV(1024,3,1)
        self.conv18 = CONV(255,1,1)

        self.conv19 = CONV(256,1,1)
        self.upsample20 = tf.keras.layers.UpSampling2D(size=[2,2])

        self.conv21 = CONV(256,1,1)
        self.conv22 = CONV(512,3,1)
        self.conv23 = CONV(256,1,1)
        self.conv24 = CONV(512,3,1)
        self.conv25 = CONV(256,1,1)
        self.conv26 = CONV(512,3,1)
        self.conv27 = CONV(255,1,1)

        self.conv28 = CONV(128,1,1)
        self.upsample29 = tf.keras.layers.UpSampling2D(size=[2,2])

        self.conv30 = CONV(128,1,1)
        self.conv31 = CONV(256,3,1)
        self.conv32 = CONV(128,1,1)
        self.conv33 = CONV(256,3,1)
        self.conv34 = CONV(128,1,1)
        self.conv35 = CONV(256,3,1)
        self.conv36 = CONV(255,1,1)




    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.blocks5(x)
        x = self.conv6(x)
        x = self.blocks7(x)
        x_52_2 = x
        x = self.conv8(x)
        x = self.blocks9(x)
        x_26_2 = x
        x = self.conv10(x)
        x = self.blocks11(x)

        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x_26_1 = x
        x = self.conv17(x)
        x = self.conv18(x)
        #   -1x13x13x255
        y_pred_13 = x

        x_26_1 = self.conv19(x_26_1)
        x_26_1 = self.upsample20(x_26_1)
        x_26 = tf.concat([x_26_1,x_26_2],axis=-1)
        x_26 = self.conv21(x_26)
        x_26 = self.conv22(x_26)
        x_26 = self.conv23(x_26)
        x_26 = self.conv24(x_26)
        x_26 = self.conv25(x_26)
        x_52_1 = x_26
        x_26 = self.conv26(x_26)
        x_26 = self.conv27(x_26)
        #   -1x26x26x255
        y_pred_26 = x_26

        x_52_1 = self.conv28(x_52_1)
        x_52_1 = self.upsample29(x_52_1)
        x_52 = tf.concat([x_52_1,x_52_2],axis=-1)
        x_52 = self.conv30(x_52)
        x_52 = self.conv31(x_52)
        x_52 = self.conv32(x_52)
        x_52 = self.conv33(x_52)
        x_52 = self.conv34(x_52)
        x_52 = self.conv35(x_52)
        x_52 = self.conv36(x_52)
        #   -1x52x52x255
        y_pred_52 = x_52


        return y_pred_13,y_pred_26,y_pred_52







