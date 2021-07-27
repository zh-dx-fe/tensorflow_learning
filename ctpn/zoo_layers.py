import tensorflow as tf

def conv_layer(inputs, params, training):
    '''define a convolutional layer with params'''
    #
    # 输入数据维度为 4-D tensor: [batch_size, width, height, channels]
    #                         or [batch_size, height, width, channels]
    #
    # params = [filters, kernel_size, strides, padding, batch_norm, relu, name]
    #
    # batch_norm = True or False
    # relu = True or False

    outputs = tf.keras.layers.Conv2D(params[0],params[1],params[2],params[3],name=params[6])(inputs)
    #
    if params[4]: # batch_norm
        outputs = tf.keras.layers.BatchNormalization(name = params[6]+'/batch_norm')(outputs,training = training)
    #
    if params[5]: # relu
        outputs = tf.keras.activations.relu(outputs, name = params[6]+'/relu')
    #
    return outputs

'''
# tf.pad(tensor, paddings, mode='CONSTANT', name=None)
#
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1,], [2, 2]].
# rank of 't' is 2.
#
# padd1 = padd_layer(conv1, [[0,0],[0,0],[0,1],[0,0]], name='padd1')
'''

def pad_layer(tensor, paddings, mode='CONSTANT', name=None):
    return tf.pad(tensor, paddings, mode)

# 最大采提层
def maxpool_layer(inputs, size, stride, padding, name):
    return tf.keras.layers.MaxPool2D(size, stride, padding, name=name)(inputs)

def block_resnet_others(inputs, layer_params, relu, training, name):
    #
    # 1，图像大小不缩小，或者，图像大小只能降，1/2, 1/3, 1/4, ...
    # 2，深度，卷积修改
    #
    with tf.name_scope(name):
        #
        #short_cut = tf.add(inputs, 0)
        short_cut = tf.identity(inputs)
        #
        shape_in = inputs.get_shape().as_list()
        #
        for item in layer_params:
            inputs = conv_layer(inputs, item, training)
        #
        shape_out = inputs.get_shape().as_list()
        #
        # 图片大小，缩小
        if shape_in[1] != shape_out[1] or shape_in[2] != shape_out[2]:
            #
            size = [shape_in[1]//shape_out[1], shape_in[2]//shape_out[2]]
            #
            short_cut = maxpool_layer(short_cut, size, size, 'valid', 'shortcut_pool')
            #
        #
        # 深度
        if shape_in[3] != shape_out[3]:
            #
            item = [shape_out[3], 1, (1,1), 'same', True, False, 'shortcut_conv']
            #
            short_cut = conv_layer(short_cut, item, training)
            #
        #
        outputs = tf.add(inputs, short_cut, name = 'add')
        #
        if relu: outputs = tf.keras.activations.relu(outputs, name = 'last_relu')
        #
    #
    return outputs


def rnn_layer(input_sequence, rnn_size):
    '''build bidirectional (concatenated output) lstm layer'''
    #
    # time_major = True
    #
    # weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.01)
    cell_fw = tf.keras.layers.LSTMCell(rnn_size)
    cell_bw = tf.keras.layers.LSTMCell(rnn_size)

    #
    # Include?
    # cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw,
    #                                         input_keep_prob=dropout_rate )
    # cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw,
    #                                         input_keep_prob=dropout_rate )
    f_rnn = tf.keras.layers.RNN(cell_fw,True,go_backwards=False)(input_sequence)
    b_rnn = tf.keras.layers.RNN(cell_bw,True,go_backwards=True)(input_sequence)
    rnn_output = tf.concat([f_rnn,b_rnn],axis=-1)


    return rnn_output  # H, W, 2*rnn_size



