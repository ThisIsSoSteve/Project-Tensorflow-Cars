import tensorflow as tf
import numpy as np
import BasicConvLSTMCell

#image size 128,72
image_width = 128
image_height = 72

output_size = 2

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, image_height, image_width, 1]) #input image data
    y = tf.placeholder(tf.float32, [None, output_size]) #labels

p_keep_hidden = tf.placeholder(tf.float32)


def myModel(X, p_keep_hidden):

    #with tf.variable_scope('conv1') as scope:
    conv1 = tf.contrib.layers.convolution2d(inputs=X,
        num_outputs = 8,
        stride=[3, 3],
        kernel_size=[6, 6],
        data_format="NHWC",
        activation_fn = tf.nn.elu,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.constant_initializer(0.1))
        #_activation_summary(conv1)


    conv2 = tf.contrib.layers.convolution2d(inputs=conv1,
        num_outputs = 8,
        stride=[2, 2],
        kernel_size=[3, 3],
        data_format="NHWC",
        activation_fn = tf.nn.elu,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.constant_initializer(0.1))


    conv2_flat = tf.contrib.layers.flatten(conv2)

    fcl1 = tf.contrib.layers.fully_connected(inputs = conv2_flat, 
        num_outputs = 128, 
        activation_fn=tf.nn.elu,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.1))

    output = tf.contrib.layers.fully_connected(inputs = fcl1, 
        num_outputs = output_size, 
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.1))

    return output
