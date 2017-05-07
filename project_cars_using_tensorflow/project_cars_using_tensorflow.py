import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

import tensorflow as tf
import numpy as np

batch_size = 128

#image size 128,72
image_width = 128
image_height = 72

output_size = 4

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        window size       window movement
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def model(X, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o):
    X = tf.reshape(X, shape=[-1, image_width, image_height, 1])

    l1 = tf.nn.relu(conv2d(X, w1) + b1)
    l1 = maxpool2d(l1)
    #l1 = tf.nn.dropout(l1, p_keep_conv)

    l2 = tf.nn.relu(conv2d(l1, w2) + b2)
    l2 = maxpool2d(l2)

    l3 = tf.nn.relu(conv2d(l3, w3) + b3)
    l3 = maxpool2d(l3)

    #fully connected
    l4 = tf.reshape(l3, [-1, 128 * 16 * 9])
    l4 =tf.nn.relu(tf.matmul(l4, w4) + b4)

    output = tf.matmul(l4, w_o) + b_o

    return output

X = tf.placeholder("float", [None, image_width * image_height]) #input image data
Y = tf.placeholder("float", [None, output_size]) #labels

w1 = init_weights([5, 5, 1, 32])       # 5x5x1 conv, 32 outputs
w2 = init_weights([5, 5, 32, 64])     # 5x5x32 conv, 64 outputs
w3 = init_weights([5, 5, 64, 128])    # 5x5x32 conv, 128 outputs
w4 = init_weights([128 * 16 * 9, 1024]) # FC 128 * 16 * 9 inputs, 1024 outputs/nodes?
w_o = init_weights([1024, output_size]) # FC 1024 inputs, 10 outputs (labels)

b1 = init_weights([32])       
b2 = init_weights([64])     
b3 = init_weights([128])    
b4 = init_weights([1024]) 
b_o = init_weights([output_size])