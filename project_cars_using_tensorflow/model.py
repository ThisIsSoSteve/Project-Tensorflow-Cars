import tensorflow as tf
import numpy as np

#image size 128,72
image_width = 128
image_height = 72

output_size = 4

x = tf.placeholder("float", [None, image_height, image_width, 1]) #input image data
y = tf.placeholder("float", [None, output_size]) #labels
z = tf.placeholder("float", [None, 1]) #speed 

p_keep_hidden = tf.placeholder("float")

def myModel(X, Z, p_keep_hidden):

    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.relu)

    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.relu)

    conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.relu)

    conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #fully connected #input = 128x72 -> l1 = 64x36 -> l2 = 32x18 -> l3 = 16x9
    l4 = tf.contrib.layers.flatten(conv3)

    l4 = tf.concat([l4, Z], 1)
    l4 = tf.reshape(l4, [-1, tf.size(l4[0])])#128 * 16 * 9 + 1])#18432

    l4 = tf.contrib.layers.fully_connected(
        inputs = l4, 
        num_outputs = 1024, 
        activation_fn=tf.nn.relu
        #biases_initializer = tf.random_normal_initializer(stddev=0.1)
        )

    l4 = tf.nn.dropout(l4, p_keep_hidden)

    output = tf.contrib.layers.fully_connected(
        inputs = l4, 
        num_outputs = output_size, 
        activation_fn=None)

    return output
