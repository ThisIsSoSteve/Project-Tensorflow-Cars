import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

#os.environ['CUDA_VISIBLE_DEVICES'] =""
import tensorflow as tf
import numpy as np
import time


model_save_path = 'E:/repos/Project-Tensorflow-Cars/project_cars_using_tensorflow/model/project_tensorflow_car_model.ckpt'


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

def model(X, w4, w_o, b4, b_o, p_keep_conv, p_keep_hidden):
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

    output = tf.matmul(l4, w_o) + b_o

    return output

x = tf.placeholder("float", [None, image_height, image_width, 1]) #input image data
y = tf.placeholder("float", [None, output_size]) #labels

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

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
                                                 capacity=(10000 + 3) * 128,
                                                 num_threads=4,
                                                 min_after_dequeue=10000)

    return images, labels

#print(image.shape)
def train_model():
    #training
    traning_path = 'project_cars_training_data.tfrecords'

    filename_queue = tf.train.string_input_producer([traning_path], num_epochs=None)
    image, label = get_training_data(filename_queue)

    #prediction = model(x, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o, p_keep_conv, p_keep_hidden)
    prediction = model(x, w4, w_o, b4, b_o, p_keep_conv, p_keep_hidden)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session() as sess:#config=config
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

    
        for i in range(200):
            x_feature, y_label = sess.run([image, label])
            #print(np.array(y_label).shape)
            #print(np.array(x_feature).shape)
            _, loss_val = sess.run([optimizer, cost], feed_dict = {x: x_feature, y: y_label, p_keep_conv: 0.8, p_keep_hidden: 0.8})
            print (loss_val)
    
        # Wait for threads to finish.
        coord.request_stop()
        coord.join(threads)
        saver.save(sess, model_save_path)
#train_model()

def countdown(count):
    while True:
        print('Countdown -', count)
        if count == 0:
            break
        count -= 1
        time.sleep(1)

def use_model():
    import grabber
    import ctypes
    import cv2
    import virtual_xbox_control as vc
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    print('Get Project Cars in focus!')
    countdown(3)
    handle = ctypes.windll.user32.GetForegroundWindow()
    grabberObject = grabber.Grabber(window=handle)

    #prediction = model(x, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o, p_keep_conv, p_keep_hidden)
    prediction = model(x, w4, w_o, b4, b_o, p_keep_conv, p_keep_hidden)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_path)
        pic = None

        while True:
            #grab the screen 
            pic = grabberObject.grab(pic)
            gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

            gray_image = cv2.resize(gray_image, (128,72))

            gray_image = np.reshape(gray_image,(1,72,128,1))

            predicted_actions = sess.run(prediction, feed_dict={x:gray_image, p_keep_conv: 1.0, p_keep_hidden: 1.0})

            #predicted_actions = predicted_actions[0]

            predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions[0]))

            #predicted_throttle_and_brakes = np.array([predicted_actions[0], predicted_actions[1]]) #throttle, brakes
            #predicted_steering = np.array([predicted_actions[2], predicted_actions[3]]) #left, right
            
            ##predicted_actions = sess.run(tf.nn.softmax(predicted_actions[0]))

            #predicted_throttle_and_brakes = sess.run(tf.nn.softmax(predicted_throttle_and_brakes))
            #predicted_steering = sess.run(tf.nn.softmax(predicted_steering))
            
            vc.control_car(predicted_actions[0], predicted_actions[1], predicted_actions[2], predicted_actions[3])
            #vc.control_car(predicted_throttle_and_brakes[0], predicted_throttle_and_brakes[1], predicted_steering[0], predicted_steering[1])
            #print(predicted_actions)

            #plt.matshow(np.reshape(gray_image[0],(72,128)), cmap=plt.cm.gray)
            #plt.show()
            #break
            time.sleep(0.1)

use_model()