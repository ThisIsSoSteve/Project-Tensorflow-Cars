import tensorflow as tf
import numpy as np

#image size 128,72
image_width = 128
image_height = 72

output_size = 4

x = tf.placeholder(tf.float32, [None, image_height, image_width, 1]) #input image data
y = tf.placeholder(tf.float32, [None, output_size]) #labels
#z = tf.placeholder(tf.float32, [None, 1]) #speed 

p_keep_hidden = tf.placeholder(tf.float32)
p_is_training = tf.placeholder(tf.bool)

def myModel(X, p_keep_hidden, p_is_training):

    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.elu)

    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #conv1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.elu)

    conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #conv2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.elu)

    conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #conv3 = tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=256,
        kernel_size=[5, 5],
        padding='same',
        activation = tf.nn.elu)

    conv4 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #conv4 = tf.contrib.layers.batch_norm(conv4, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    #fully connected #input = 128x72 -> l1 = 64x36 -> l2 = 32x18 -> l3 = 16x9
    l4 = tf.contrib.layers.flatten(conv4)

    #l4 = tf.concat([l4, Z], 1)
    #l4 = tf.reshape(l4, [-1, tf.size(l4[0])])

    l4 = tf.contrib.layers.fully_connected(
        inputs = l4, 
        num_outputs = 256, 
        activation_fn=tf.nn.elu,
        biases_initializer = tf.random_normal_initializer(stddev=0.1)
        )

    #l4 = tf.contrib.layers.batch_norm(l4, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    l4 = tf.nn.dropout(l4, p_keep_hidden)

    l5 = tf.contrib.layers.fully_connected(
        inputs = l4, 
        num_outputs = 128, 
        activation_fn=tf.nn.elu,
        biases_initializer = tf.random_normal_initializer(stddev=0.1)
        )

    #l5 = tf.contrib.layers.batch_norm(l5, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

    l5 = tf.nn.dropout(l5, p_keep_hidden)


    output = tf.contrib.layers.fully_connected(
        inputs = l5, 
        num_outputs = output_size, 
        activation_fn=None)

    return output
