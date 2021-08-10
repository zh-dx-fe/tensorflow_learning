import tensorflow as tf
import zoo_layers as layers




def crnn(input, nh, n_class):
    # [1, 32, W, 3]
    params = [
        [64, 3, 1, 'same', False, True, 'conv_1'],
        [128, 3, 1, 'same', False, True, 'conv_2'],
        [256, 3, 1, 'same', True, True, 'conv_3'],
        [256, 3, 1, 'same', False, True, 'conv_4'],
        [512, 3, 1, 'same', True, True, 'conv_5'],
        [512, 3, 1, 'same', False, True, 'conv_6'],
        [512, 2, 1, 'valid', True, True, 'conv_7'],
              ]
    x = layers.conv_layer(input, params=params[0])  # [1, 32, W, 64]
    x = layers.maxpool_layer(x, 2, 2, padding='valid', name='pool_1')  # [1, 16, W/2, 64]
    x = layers.conv_layer(x, params=params[1])  # [1, 16, W/2, 128]
    x = layers.maxpool_layer(x, 2, 2, padding='valid', name='pool_2')  # [1, 8, W/4, 128]
    x = layers.conv_layer(x, params=params[2])  # [1, 8, W/4, 256]
    x = layers.conv_layer(x, params=params[3])  # [1, 8, W/4, 256]
    x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0]])  # [1, 8, W/4+1, 256]
    x = layers.maxpool_layer(x, (2, 2), (2, 1), 'valid', name='pool_3')  # [1, 4, W/4, 256]
    x = layers.conv_layer(x, params=params[4])  # [1, 4, W/4, 512]
    x = layers.conv_layer(x, params=params[5])  # [1, 4, W/4, 512]
    x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0]])  # [1, 4, W/4+1, 512]
    x = layers.maxpool_layer(x, (2, 2), (2, 1), 'valid', name='pool_4') # [1, 2, W/4, 512]
    x = layers.conv_layer(x, params=params[6])  # [1, 1, W/4, 512]
    x = tf.squeeze(x, axis=0)  # [1, W/4, 512]
    x = layers.rnn_layer(x, nh)  # [1, W/4, nh*2]
    x = tf.keras.layers.Dense(nh)(x)  # [1, W/4, nh]
    x = layers.rnn_layer(x, nh)  # [1, W/4, nh*2]
    x = tf.keras.layers.Dense(n_class)(x)  # [1, W/4, n_class]
    return x














