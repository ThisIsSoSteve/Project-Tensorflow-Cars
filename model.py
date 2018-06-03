import tensorflow as tf
from lazy_property import lazy_property

class Model:


    def __init__(self, feature, label, learning_rate, conv_keep_prob, dense_keep_prob):
        self.feature = feature
        self.label = label
        self.learning_rate = learning_rate
        self.conv_keep_prob = conv_keep_prob
        self.dense_keep_prob = dense_keep_prob
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):

        conv1 = tf.contrib.layers.convolution2d(
            inputs=self.feature,
            num_outputs = 24,
            stride=[2, 2],
            kernel_size=[6, 6],
            padding = 'VALID',
            data_format='NHWC',
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))
    
        conv1_drop_out = tf.nn.dropout(conv1, self.conv_keep_prob) 

        conv2 = tf.contrib.layers.convolution2d(
            inputs=conv1_drop_out,
            num_outputs = 36,
            stride=[2, 2],
            kernel_size=[5, 5],
            padding = 'SAME',
            data_format="NHWC",
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))
        
        conv2_drop_out = tf.nn.dropout(conv2, self.conv_keep_prob) 

        conv3 = tf.contrib.layers.convolution2d(
            inputs=conv2_drop_out,
            num_outputs = 48,
            stride=[2, 2],
            kernel_size=[5, 5],
            padding = 'SAME',
            data_format="NHWC",
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))

        conv3_drop_out = tf.nn.dropout(conv3, self.conv_keep_prob) 

        conv4 = tf.contrib.layers.convolution2d(
            inputs=conv3_drop_out,
            num_outputs = 64,
            stride=[1, 1],
            kernel_size=[3, 3],
            padding = 'SAME',
            data_format="NHWC",
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))

        conv4_drop_out = tf.nn.dropout(conv4, self.conv_keep_prob) 

        conv5 = tf.contrib.layers.convolution2d(
            inputs=conv4_drop_out,
            num_outputs = 64,
            stride=[1, 1],
            kernel_size=[3, 3],
            padding = 'SAME',
            data_format="NHWC",
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.constant_initializer(0.1))

        conv5_drop_out = tf.nn.dropout(conv5, self.conv_keep_prob) 


        conv5_flat = tf.contrib.layers.flatten(conv5_drop_out)

        fcl1 = tf.contrib.layers.fully_connected(
            inputs = conv5_flat, 
            num_outputs = 128, 
            activation_fn = tf.nn.elu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1)
            )

        fcl1_drop_out = tf.nn.dropout(fcl1, self.dense_keep_prob) 

        fcl2 = tf.contrib.layers.fully_connected(
            inputs = fcl1_drop_out, 
            num_outputs = 64, 
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1)
            )
        
        fcl2_drop_out = tf.nn.dropout(fcl2, self.dense_keep_prob) 

        fcl3 = tf.contrib.layers.fully_connected(
            inputs = fcl2_drop_out, 
            num_outputs = 16, 
            activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1)
            )

        fcl3_drop_out = tf.nn.dropout(fcl3, self.dense_keep_prob) 

        output = tf.contrib.layers.fully_connected(
            inputs = fcl3, 
            num_outputs = 3, 
            activation_fn = None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1)
            )

        return output

    @lazy_property
    def optimize(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label))
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost), cost
        #return tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost), cost

    @lazy_property
    def error(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#image size 128,72
#image_width = 32
#image_height = 18

#output_size = 3

#x = tf.placeholder(tf.float32, [None, image_height, image_width, 1]) #input image data
#y = tf.placeholder(tf.float32, [None, output_size]) #labels
#z = tf.placeholder(tf.float32, [None, 1]) #speed 

#p_keep_hidden = tf.placeholder(tf.float32)
#batch_size = tf.placeholder(tf.int32)

# def myModel(X, p_keep_hidden):

#     conv1 = tf.contrib.layers.convolution2d(
#         inputs=X,
#         num_outputs = 8,
#         stride=[2, 2],
#         kernel_size=[6, 6],
#         data_format="NHWC",
#         activation_fn = tf.nn.relu,
#         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#         biases_initializer=tf.constant_initializer(0.1))
    

#     conv2 = tf.contrib.layers.convolution2d(
#         inputs=conv1,
#         num_outputs = 8,
#         stride=[2, 2],
#         kernel_size=[3, 3],
#         data_format="NHWC",
#         activation_fn = tf.nn.relu,
#         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#         biases_initializer=tf.constant_initializer(0.1))

#     conv2_flat = tf.contrib.layers.flatten(conv2)

#     #l4 = tf.concat([l4, Z], 1)
#     #l4 = tf.reshape(l4, [-1, tf.size(l4[0])])

#     fcl1 = tf.contrib.layers.fully_connected(
#         inputs = conv2_flat, 
#         num_outputs = 128, 
#         activation_fn=tf.nn.elu,
#         weights_initializer=tf.contrib.layers.xavier_initializer(),
#         biases_initializer=tf.constant_initializer(0.1)
#         )

#     #l4 = tf.contrib.layers.batch_norm(l4, center=True, scale=True, is_training=p_is_training, activation_fn = tf.nn.relu)

#     #fcl1 = tf.nn.dropout(fcl1, p_keep_hidden)

#     fcl2 = tf.contrib.layers.fully_connected(
#         inputs = fcl1, 
#         num_outputs = 32, 
#         activation_fn=tf.nn.relu,
#         weights_initializer=tf.contrib.layers.xavier_initializer(),
#         biases_initializer=tf.constant_initializer(0.1)
#         )

#     output = tf.contrib.layers.fully_connected(
#         inputs = fcl2, 
#         num_outputs = output_size, 
#         activation_fn=None,
#         weights_initializer=tf.contrib.layers.xavier_initializer(),
#         biases_initializer=tf.constant_initializer(0.1)
#         )

#     return output
