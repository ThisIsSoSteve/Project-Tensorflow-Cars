import tensorflow as tf
import numpy as np

#image size 128,72
image_width = 32
image_height = 18

output_size = 3

x = tf.placeholder(tf.float32, [None, image_height, image_width, 1]) #input image data
y = tf.placeholder(tf.float32, [None, output_size]) #labels
#z = tf.placeholder(tf.float32, [None, 1]) #speed 

p_keep_hidden = tf.placeholder(tf.float32)
#batch_size = tf.placeholder(tf.int32)

def myModel(X, p_keep_hidden):

    conv1 = tf.contrib.layers.convolution2d(
        inputs=X,
        num_outputs = 8,
        stride=[3, 3],
        kernel_size=[6, 6],
        data_format="NHWC",
        activation_fn = tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.constant_initializer(0.01))
    

    conv2 = tf.contrib.layers.convolution2d(
        inputs=conv1,
        num_outputs = 8,
        stride=[2, 2],
        kernel_size=[3, 3],
        data_format="NHWC",
        activation_fn = tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.constant_initializer(0.01))


    # with tf.variable_scope('conv2_lstm2', initializer = tf.constant_initializer(0.1)):#tf.random_uniform_initializer(-.01, 0.1)):
    #   cell = BasicConvLSTMCell.BasicConvLSTMCell([12,22], [2,2], 8, activation=tf.nn.elu)
    #   #if hidden is None:
    #   hidden = cell.zero_state(batch_size, tf.int32) 
    #   convLstm1, hidden = cell(conv2, hidden)

    #conv3 = tf.contrib.layers.convolution2d(
    #    inputs=conv2,
    #    num_outputs = 8,
    #    stride=[3, 3],
    #    kernel_size=[2, 2],
    #    data_format="NHWC",
    #    activation_fn = tf.nn.elu,
    #    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #    biases_initializer=tf.constant_initializer(0.1))

    #conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #conv1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #conv2 = tf.layers.conv2d(
    #    inputs=conv1,
    #    filters=36,
    #    kernel_size=[5, 5],
    #    padding='same',
    #    activation = tf.nn.elu)

   # conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #conv2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #conv3 = tf.layers.conv2d(
    #    inputs=conv2,
    #    filters=48,
    #    kernel_size=[5, 5],
    #    padding='same',
    #    activation = tf.nn.elu)

    #conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #conv3 = tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #conv4 = tf.layers.conv2d(
    #    inputs=conv3,
    #    filters=64,
    #    kernel_size=[3, 3],
    #    padding='same',
    #    activation = tf.nn.elu)

    #conv4 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=1)

    ##conv4 = tf.contrib.layers.batch_norm(conv4, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #conv5 = tf.layers.conv2d(
    #    inputs=conv4,
    #    filters=64,
    #    kernel_size=[3, 3],
    #    padding='same',
    #    activation = tf.nn.elu)

    #conv5 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=1)

    ##conv4 = tf.contrib.layers.batch_norm(conv4, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #fully connected #input = 128x72 -> l1 = 64x36 -> l2 = 32x18 -> l3 = 16x9
    conv2_flat = tf.contrib.layers.flatten(conv2)

    #l4 = tf.concat([l4, Z], 1)
    #l4 = tf.reshape(l4, [-1, tf.size(l4[0])])

    fcl1 = tf.contrib.layers.fully_connected(
        inputs = conv2_flat, 
        num_outputs = 128, 
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.01)
        )

    #l4 = tf.contrib.layers.batch_norm(l4, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #fcl1 = tf.nn.dropout(fcl1, p_keep_hidden)

    fcl2 = tf.contrib.layers.fully_connected(
        inputs = fcl1, 
        num_outputs = 32, 
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.01)
        )

    #fcl2 = tf.nn.dropout(fcl2, p_keep_hidden)

    #fcl3 = tf.contrib.layers.fully_connected(
    #    inputs = fcl2, 
    #    num_outputs = 50, 
    #    activation_fn=tf.nn.elu,
    #    biases_initializer = tf.random_normal_initializer(stddev=0.1)
    #    )

    #fcl3 = tf.nn.dropout(fcl3, p_keep_hidden)

    #fcl4 = tf.contrib.layers.fully_connected(
    #    inputs = fcl3, 
    #    num_outputs = 10, 
    #    activation_fn=tf.nn.elu,
    #    biases_initializer = tf.random_normal_initializer(stddev=0.1)
    #    )

    #fcl4 = tf.nn.dropout(fcl4, p_keep_hidden)

    output = tf.contrib.layers.fully_connected(
        inputs = fcl2, 
        num_outputs = output_size, 
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.01)
        )

    return output
