import tensorflow as tf
import numpy as np

def conv3x3(channels,stride=1,kernel=(3,3)):
    return tf.keras.layers.Conv2D(channels,kernel,strides=stride,padding='same',use_bias=False,kernel_initializer=tf.random_normal_initializer())

class ResnetBlock(tf.keras.Model):
    def __init__(self,channels,strides=1,residual_path=False,):
        super(ResnetBlock,self).__init__()
        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv1 = conv3x3(channels,strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels,strides,kernel=(1,1))
            self.down_bn = tf.keras.layers.BatchNormalization()
    def call(self,inputs,training=None):
        residuals = inputs

        x = self.bn1(inputs,training = training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x,training = training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residuals = self.down_bn(inputs,training=training)
            residuals = tf.nn.relu(residuals)
            residuals = self.down_conv(residuals)

        x = x + residuals
        return x
class ResNet(tf.keras.Model):
    def __init__(self,block_list,num_classes,initial_filters=16,**kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)
        self.blocks = tf.keras.models.Sequential(name='dynamic-blocks')

        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id !=0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels,strides=2,residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels,residual_path=residual_path)

                self.in_channels = self.out_channels
                self.blocks.add(block)
            self.out_channels *= 2
        self.final_bn = tf.keras.layers.BatchNormalization()
        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.fc = tf.keras.layers.Dense(num_classes,activation='softmax')

    def call(self, inputs, training=None):
        out = self.conv_initial(inputs)
        out = self.blocks(out,training = training)
        out = self.final_bn(out,training = training)
        out = tf.nn.relu(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        # out = tf.nn.softmax(out)

        return out

cifar = tf.keras.datasets.cifar10
data = cifar.load_data()
(train_x,train_y),(test_x,test_y) = data
train_x,test_x = tf.cast(train_x,dtype=tf.float32)/255.0 , tf.cast(test_x,dtype=tf.float32)/255.0
train_y,test_y = tf.one_hot(train_y,depth=10) , tf.one_hot(test_y,depth=10)
train_y,test_y = tf.reshape(train_y,(50000,10)), tf.reshape(test_y,(10000,10))
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

def main():
    model = ResNet([2,2,2],num_classes=10)
    model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    model.build(input_shape=(None,32,32,3))
    model.summary()
    # model.fit(train_x,train_y,batch_size=32,epochs=10,validation_data=(test_x,test_y),verbose=1)

main()