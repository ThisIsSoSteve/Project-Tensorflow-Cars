import tensorflow as tf
import numpy as np
import BasicConvLSTMCell

#image size 128,72
image_width = 640
image_height = 360

output_size = 2

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, image_height, image_width, 1]) #input image data
    y = tf.placeholder(tf.float32, [None, output_size]) #labels

p_keep_hidden = tf.placeholder(tf.float32)


def converlution_visualiser(convulation_data, name):
    with tf.variable_scope(name + '_visualisation'):
        conv_shape = convulation_data.get_shape().as_list()
        #print(conv1_shape)
        #print(conv1_shape[3])
        channels = conv_shape[3]
        iy = conv_shape[1]
        ix = conv_shape[2]
        V = tf.slice(convulation_data,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
        V = tf.reshape(V,(iy,ix,channels))
        #Next add a couple of pixels of zero padding around the image
        ix += 4
        iy += 4
        V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)

        #Layout in a grid of 4 by 2
        cy = 2
        cx = 4
        V = tf.reshape(V,(iy,ix,cy,cx)) 
        V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix

        V = tf.reshape(V,(1,cy*iy,cx*ix,1))
        # this will display grid created
        tf.summary.image(name + '/filters', V, max_outputs=1, collections=None)

def myModel(X, p_keep_hidden):

    #Multidimensional discrete convolution
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.contrib.layers.convolution2d(inputs=X,
            num_outputs = 8,
            stride=[4, 4],
            kernel_size=[8, 8],
            data_format="NHWC",
            activation_fn = tf.nn.elu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))

    converlution_visualiser(conv1, 'conv1')

    with tf.variable_scope('conv2') as scope:
        conv2 = tf.contrib.layers.convolution2d(inputs=conv1,
            num_outputs = 8,
            stride=[2, 2],
            kernel_size=[4, 4],
            data_format="NHWC",
            activation_fn = tf.nn.elu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))

    converlution_visualiser(conv2, 'conv2')

    ##Learns a bit faster with a 3rd conv
    with tf.variable_scope('conv3') as scope:
        conv3 = tf.contrib.layers.convolution2d(inputs=conv2,
            num_outputs = 8,
            stride=[2, 2],
            kernel_size=[2, 2],
            data_format="NHWC",
            activation_fn = tf.nn.elu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))

    converlution_visualiser(conv3, 'conv3')
    

    conv2_flat = tf.contrib.layers.flatten(conv3)

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


