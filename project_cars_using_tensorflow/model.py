import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        window size       window movement
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Setup Variables 

#image size 128,72
image_width = 128
image_height = 72

output_size = 4

x = tf.placeholder("float", [None, image_height, image_width, 1]) #input image data
y = tf.placeholder("float", [None, output_size]) #labels

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

#w1 = init_weights([5, 5, 1, 32])       # 5x5x1 conv, 32 outputs
#w2 = init_weights([5, 5, 32, 64])     # 5x5x32 conv, 64 outputs
#w3 = init_weights([5, 5, 64, 128])    # 5x5x32 conv, 128 outputs
w4 = init_weights([128 * 16 * 9, 1024]) # FC 128 * 16 * 9 inputs, 1024 outputs/nodes?
w5 = init_weights([1024, 1024])
w_o = init_weights([1024, output_size]) # FC 1024 inputs, 10 outputs (labels)

#b1 = init_weights([32])       
#b2 = init_weights([64])     
#b3 = init_weights([128])    
b4 = init_biases([1024])
b5 = init_biases([1024])  
b_o = init_biases([output_size])


def myModel(X, p_keep_hidden):
    #X = tf.reshape(X, shape=[-1, image_width, image_height, 1])
    
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

    #l1 = tf.nn.relu(conv2d(X, w1) + b1)
    #l1 = maxpool2d(l1)
    #l1 = tf.nn.dropout(l1, p_keep_conv)

    #l2 = tf.nn.relu(conv2d(l1, w2) + b2)
    #l2 = maxpool2d(l2)
    #l2 = tf.nn.dropout(l2, p_keep_conv)

    #l3 = tf.nn.relu(conv2d(l2, w3) + b3)
    #l3 = maxpool2d(l3)
    #l3 = tf.nn.dropout(l3, p_keep_conv)

    #fully connected #input = 128x72 -> l1 = 64x36 -> l2 = 32x18 -> l3 = 16x9
    l4 = tf.reshape(conv3, [-1, 128 * 16 * 9]) #output_number * layer_size #73728
    l4 =tf.nn.relu(tf.matmul(l4, w4) + b4)
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    #l5 = tf.nn.relu(tf.matmul(l4, w5)+ b5)
    #l5 = tf.nn.dropout(l5, p_keep_hidden)

    output = tf.matmul(l4, w_o) + b_o

    return output
