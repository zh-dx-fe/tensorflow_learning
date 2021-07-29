import tensorflow as tf
import zoo_layers as layers


def conv_feat_layers(inputs, width, training):
    #
    # convolutional features maps for detection
    #

    #
    # detection
    #
    # [3,1; 1,1],
    # [9,2; 3,2], [9,2; 3,2], [9,2; 3,2]
    # [18,4; 6,4], [18,4; 6,4], [18,4; 6,4]
    # [36,8; 12,8], [36,8; 12,8], [36,8; 12,8],
    #
    # anchor width:  8,
    # anchor height: 6, 12, 24, 36,
    #
    # feature_layer --> receptive_field
    # [0,0] --> [0:36, 0:8]
    # [0,1] --> [0:36, 8:8+8]
    # [i,j] --> [12*i:36+12*i, 8*j:8+8*j]
    #
    # feature_layer --> anchor_center
    # [0,0] --> [18, 4]
    # [0,1] --> [18, 4+8]
    # [i,j] --> [18+12*i, 4+8*j]
    #

    #
    layer_params = [[64, (3, 3), (1, 1), 'same', True, True, 'conv1'],
                    [128, (3, 3), (1, 1), 'same', True, True, 'conv2'],
                    [128, (2, 2), (2, 2), 'valid', True, True, 'pool1'],  # for pool
                    [128, (3, 3), (1, 1), 'same', True, True, 'conv3'],
                    [256, (3, 3), (1, 1), 'same', True, True, 'conv4'],
                    [256, (2, 2), (2, 2), 'valid', True, True, 'pool2'],  # for pool
                    [256, (3, 3), (1, 1), 'same', True, True, 'conv5'],
                    [512, (3, 3), (1, 1), 'same', True, True, 'conv6'],
                    [512, (3, 2), (3, 2), 'valid', True, True, 'pool3'],  # for pool
                    [512, (3, 1), (1, 1), 'valid', True, True, 'conv_feat']]  # for feat

    #
    with tf.name_scope("conv_comm"):
        #
        inputs = layers.conv_layer(inputs, layer_params[0], training)
        inputs = layers.conv_layer(inputs, layer_params[1], training)
        inputs = layers.pad_layer(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], name='padd1')
        # inputs = layers.conv_layer(inputs, layer_params[2], training)
        inputs = layers.maxpool_layer(inputs, (2, 2), (2, 2), 'valid', 'pool1')
        #
        params = [[128, 3, (1, 1), 'same', True, True, 'res1/conv1'],
                  [128, 3, (1, 1), 'same', True, False, 'res1/conv2']]
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res1')
        #
        inputs = layers.conv_layer(inputs, layer_params[3], training)
        inputs = layers.conv_layer(inputs, layer_params[4], training)
        inputs = layers.pad_layer(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], name='padd2')
        # inputs = layers.conv_layer(inputs, layer_params[5], training)
        inputs = layers.maxpool_layer(inputs, (2, 2), (2, 2), 'valid', 'pool2')
        #
        params = [[256, 3, (1, 1), 'same', True, True, 'res2/conv1'],
                  [256, 3, (1, 1), 'same', True, False, 'res2/conv2']]
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res2')
        #
        inputs = layers.conv_layer(inputs, layer_params[6], training)
        inputs = layers.conv_layer(inputs, layer_params[7], training)
        inputs = layers.pad_layer(inputs, [[0, 0], [0, 0], [0, 1], [0, 0]], name='padd3')
        inputs = layers.conv_layer(inputs, layer_params[8], training)
        # inputs = layers.maxpool_layer(inputs, (3,2), (3,2), 'valid', 'pool3')
        #
        params = [[512, 3, (1, 1), 'same', True, True, 'res3/conv1'],
                  [512, 3, (1, 1), 'same', True, False, 'res3/conv2']]
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res3')
        #
        conv_feat = layers.conv_layer(inputs, layer_params[9], training)
        #
        feat_size = tf.shape(conv_feat)
        #
    #
    # Calculate resulting sequence length from original image widths
    #
    # two = tf.constant(2, dtype=tf.float32, name='two')
    # #
    # w = tf.cast(width, tf.float32)
    # #
    # w = tf.math.divide(w, two)
    # w = tf.math.ceil(w)
    # #
    # w = tf.math.divide(w, two)
    # w = tf.math.ceil(w)
    # #
    # w = tf.math.divide(w, two)
    # w = tf.math.ceil(w)
    # #
    # w = tf.cast(w, tf.int32)
    # #
    # w = tf.tile(w, [feat_size[1]])
    # #
    # # Vectorize
    # sequence_length = tf.reshape(w, [-1], name='seq_len')
    #

    #
    return conv_feat  # N, H, W, 512


def rnn_detect_layers(conv_feat, num_anchors):
    #
    # one-picture features
    conv_feat = tf.squeeze(conv_feat, axis=0)  # squeeze # (32, W, 512)
    #
    #
    # Transpose to time-major order for efficiency
    #  --> [paddedSeqLen batchSize numFeatures]
    #
    # rnn_sequence = tf.transpose(conv_feat, perm=[1, 0, 2], name='time_major')# (W, 32, 512)
    rnn_sequence = conv_feat

    #
    rnn_size = 256  # 256, 512
    fc_size = 512  # 256, 384, 512
    #
    #
    rnn1 = layers.rnn_layer(rnn_sequence, rnn_size)
    rnn2 = layers.rnn_layer(rnn1, rnn_size)
    # rnn3 = rnn_layer(rnn2, sequence_length, rnn_size, 'bdrnn3')
    #
    weight_initializer = tf.keras.initializers.variance_scaling()

    bias_initializer = tf.keras.initializers.constant(value=0.0)
    #
    rnn_feat = tf.keras.layers.Dense(fc_size,tf.keras.activations.relu,
                                     kernel_initializer=weight_initializer,bias_initializer=bias_initializer,name='rnn_feat')(rnn2)

    rnn_cls = tf.keras.layers.Dense(num_anchors * 2,
                                    activation=tf.keras.activations.sigmoid,
                                    kernel_initializer=weight_initializer,bias_initializer=bias_initializer,name='text_cls')(rnn_feat)

    rnn_ver = tf.keras.layers.Dense(num_anchors * 2,
                                    activation=tf.keras.activations.tanh,
                                    kernel_initializer=weight_initializer,bias_initializer=bias_initializer,name='text_ver')(rnn_feat)

    rnn_hor = tf.keras.layers.Dense(num_anchors * 2,
                                    activation=tf.keras.activations.tanh,
                                    kernel_initializer=weight_initializer,bias_initializer=bias_initializer,name='text_hor')(rnn_feat)

    # (32, w, 2k)
    return rnn_cls, rnn_ver, rnn_hor


def detect_loss(rnn_cls, rnn_ver, rnn_hor, target_cls, target_ver, target_hor):

    # loss_cls
    cls_pred_shape = tf.shape(rnn_cls)  # cls_pred shape(32,51,2A) ->shape(4,?)
    cls_pred_reshape = tf.reshape(rnn_cls, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])  # shape(32,51,A,2)
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])  # (32*51*A, 2)
    cls_true_reshape = tf.reshape(target_cls, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])  # shape(32,51,A,2)
    true_cls_score = tf.reshape(cls_true_reshape, [-1, 2])  # (32*51*A, 2)
    mask = tf.reduce_sum(true_cls_score,axis=-1) == 1
    mask = mask.numpy()
    loss_cls = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(true_cls_score[mask],rpn_cls_score[mask]))

    # loss reg

    mask_reg = tf.equal(true_cls_score[:,0],1)

    ver_true = tf.reshape(target_ver, [32,51, -1, 2])  # shape(32,51,A,2)
    hor_true = tf.reshape(target_hor, [32,51, -1, 2])  # shape(32,51,A,2)
    ver_true = tf.reshape(ver_true, [-1, 2])  # (32*51*A, 2)
    hor_true = tf.reshape(hor_true, [-1, 2])  # (32*51*A, 2)

    ver_pred = tf.reshape(rnn_ver, [32, 51, -1, 2])  # shape(32,51,A,2)
    hor_pred = tf.reshape(rnn_hor, [32, 51, -1, 2])  # shape(32,51,A,2)
    ver_pred = tf.reshape(ver_pred, [-1, 2])  # (32*51*A, 2)
    hor_pred = tf.reshape(hor_pred, [-1, 2])  # (32*51*A, 2)

    delta_ver = tf.cast(ver_true[mask_reg],tf.float32) - tf.cast(ver_pred[mask_reg],tf.float32)
    delta_hor = tf.cast(hor_true[mask_reg],tf.float32) - tf.cast(hor_pred[mask_reg],tf.float32)
    smooth_ver = tf.reduce_mean(smooth_l1_dist(delta_ver))
    smooth_hor = tf.reduce_mean(smooth_l1_dist(delta_hor))

    loss = smooth_hor + smooth_ver + loss_cls
    return loss

def smooth_l1_dist(deltas, sigma2=9.0):
    deltas_abs = tf.abs(deltas)
    smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
    return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)