import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

dtype = tf.float32

BAND_NUM_30 = 5
BAND_NUM_15 = 1




def vgg_model(inputs):

    RGB, rest_30, bands_15 = tf.split(inputs, [3, 5, 1], axis=-1)

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        #===================================================================================
        #                               first block
        #===================================================================================
        with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
            kernel_rgb = tf.get_variable('kernel_rgb', [3,3,3,64], dtype=dtype)
            kernel_15 = tf.get_variable('kernel_15', [3,3,BAND_NUM_15,64], dtype=dtype)
            kernel_30 = tf.get_variable('kernel_30', [3,3,BAND_NUM_30,64], dtype=dtype)
            biases = tf.get_variable('biases_1', [64], dtype=dtype)

            fea_map_rgb = tf.nn.atrous_conv2d(RGB, kernel_rgb, rate=2, padding='SAME')
            fea_map_30 = tf.nn.atrous_conv2d(rest_30, kernel_30, rate=2, padding='SAME')
            fea_map_15 = tf.nn.conv2d(bands_15, kernel_15, [1,1,1,1], padding='SAME')

            fea_map = fea_map_rgb + fea_map_30 + fea_map_15
            net = tf.nn.relu(tf.nn.bias_add(fea_map,biases))

            kernel_2 = tf.get_variable('weights_2', [3,3,64,64], dtype=dtype)
            biases_2 = tf.get_variable('biases_2', [64], dtype=dtype)
            net = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(net,kernel_2,[1,1,1,1],'SAME'),biases_2))
        # ===================================================================================
        #                               rest blocks
        # ===================================================================================
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.conv2d(net, 4096, [7,7], padding='VALID', scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.conv2d(net, 4096, [1,1], padding='VALID', scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.conv2d(net, 3, [1,1], padding='VALID', scope='fc8')
        net = tf.layers.flatten(net)
    return net

























































