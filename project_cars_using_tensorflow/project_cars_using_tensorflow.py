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

    l3 = tf.nn.relu(conv2d(l2, w3) + b3)
    l3 = maxpool2d(l3)

    #fully connected
    l4 = tf.reshape(l3, [-1, 128 * 16 * 9])
    l4 =tf.nn.relu(tf.matmul(l4, w4) + b4)

    output = tf.matmul(l4, w_o) + b_o

    return output

x = tf.placeholder("float", [None, image_height, image_width, 1]) #input image data
y = tf.placeholder("float", [None, output_size]) #labels

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

#read and decode training data
def get_training_data(filename_queue):
    reader = tf.TFRecordReader()

    _, data = reader.read(filename_queue)
    features = tf.parse_single_example(
      data,
      features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([4], tf.float32)
        })
   
    #print('image', features['image'])
    image = tf.decode_raw(features['image'], tf.uint8)
    image_shape = tf.stack([image_height, image_width, 1])#raw shape has to be defined
    image = tf.reshape(image, image_shape)

    label = features['label']

    #print('label', label)
    #image = np.fromstring(features['image'], dtype=np.uint8)
    #image = np.reshape(test, (72, 128))

    images, labels = tf.train.shuffle_batch([image, label],
                                                 batch_size=128,
                                                 capacity=(50 + 3) * 128,
                                                 num_threads=2,
                                                 min_after_dequeue=50)

    return images, labels

#training
traning_path = 'project_cars_training_data.tfrecords'

filename_queue = tf.train.string_input_producer([traning_path], num_epochs=10)
image, label = get_training_data(filename_queue)

#print(image.shape)

prediction = model(x, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    
    for i in range(10):
        x_feature, y_label = sess.run([image, label])
        #print(np.array(y_label).shape)
        #print(np.array(x_feature).shape)
        _, loss_val = sess.run([optimizer, cost], feed_dict = {x: x_feature, y: y_label})
        print (loss_val)
    
    # Wait for threads to finish.
    coord.request_stop()
    coord.join(threads)
    