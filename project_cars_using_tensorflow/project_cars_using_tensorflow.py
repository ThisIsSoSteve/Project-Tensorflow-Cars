import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

#os.environ['CUDA_VISIBLE_DEVICES'] =""
import tensorflow as tf
import numpy as np
import time
import model


model_save_path = 'E:/repos/Project-Tensorflow-Cars/project_cars_using_tensorflow/model/project_tensorflow_car_model.ckpt'

batch_size = 128
epochs = 1
#path = 'data/Project_Cars_2017-05-23_22-23-00/'
def train_model():
    #load data
    training_path = 'training_full.npy'#'training_balance_data.npy'
    training_data = np.load(training_path)

    prediction = model.myModel(model.x, model.p_keep_hidden)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=model.y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            np.random.shuffle(training_data)
            
            train_x = []
            train_y = []
            for data in training_data:#better way?
                train_x.append(np.array(data[0]))
                train_y.append(np.array(data[1]))

            while i < len(training_data[0]):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                #batch_x = sess.run(tf.reshape(batch_x, shape=[-1,image_height, image_width, 1]))
                batch_x = batch_x.reshape((-1, model.image_height, model.image_width, 1))
                #print(np.array(batch_y).shape)
                #print(np.array(batch_x).shape)
                _, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.p_keep_hidden: 0.8})
                epoch_loss += loss_val
                i += batch_size
            print ("Epoch:", epoch + 1, 'Loss:',  epoch_loss)
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

    controller = vc.virtual_xbox_controller()

    print('Get Project Cars in focus!')
    countdown(3)
    handle = ctypes.windll.user32.GetForegroundWindow()
    grabberObject = grabber.Grabber(window=handle)

    #prediction = model(x, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o, p_keep_conv, p_keep_hidden)
    prediction = model.myModel(model.x, model.p_keep_hidden)
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

            predicted_actions = sess.run(prediction, feed_dict={model.x:gray_image, model.p_keep_hidden: 1.0})

            predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions[0]))

            controller.control_car(predicted_actions[0], predicted_actions[1], predicted_actions[2], predicted_actions[3])
            
            #print("Throttle:", predicted_actions[0], "Brakes:", predicted_actions[1], "left:", predicted_actions[2], 'right', predicted_actions[3])

            #print(predicted_actions)

            #plt.matshow(np.reshape(gray_image[0],(72,128)), cmap=plt.cm.gray)
            #plt.show()
            #break
            time.sleep(0.1)

use_model()

