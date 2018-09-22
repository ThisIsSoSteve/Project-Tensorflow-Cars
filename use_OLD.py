import time
from common import countdown
import numpy as np

import tensorflow as tf
import model
from common import grabber
import ctypes
import cv2
import virtual_xbox_control as vc
import carseour as pcars
from model import Model

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
class Use_Old:

    def __init__(self, image_height, image_width, output_size):
        self.image_height = image_height
        self.image_width = image_width
        self.output_size = output_size

        self.X = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.output_size])
        self.conv_keep_rate = tf.placeholder(tf.float32)
        self.dense_keep_rate = tf.placeholder(tf.float32)
        self.model = Model(self.X, self.Y, 0.01, self.conv_keep_rate, self.dense_keep_rate)

    def use_model(self, checkpoint_save_path):
        controller = vc.virtual_xbox_controller()

        print('Get Project Cars in focus!')
        countdown.begin_from(3)
        #handle = ctypes.windll.user32.GetForegroundWindow()
        #grabberObject = grabber.Grabber(window=handle)
        grabber1 = grabber.Grabber(window_title='Project CARS™')

        #prediction = model.myModel(model.x, model.p_keep_hidden, model.batch_size)# model.z
        
        saver = tf.train.Saver()

        game_running = False
        cropped_pixels = int((self.image_width - self.image_height) / 2)
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:#config=config
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_save_path)
            pic = None
            game = pcars.live()

            while True:
                
                if game.mGameState == 2:
                    game_running = True
                    #gameSpeed = np.array([game.mSpeed])
                    #gameSpeed = np.reshape(gameSpeed, (1, 1))

                    pic = grabber1.grab()#grabberObject.grab(pic)#grab the screen 
                    gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                    gray_image = cv2.resize(gray_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
                    gray_image = np.float16(gray_image / 255.0)

                    #gray_image = gray_image[0:self.image_height, cropped_pixels: cropped_pixels + self.image_height]
                    #gray_image = np.reshape(gray_image, (-1, self.image_height, self.image_width, 1)) 
                    #gray_image[gray_image == 0] = -1.0

                    ##predicted_actions = sess.run(prediction, feed_dict={model.x:gray_image, model.z:gameSpeed, model.p_keep_hidden: 1.0})
                    ##predicted_actions = sess.run(prediction, feed_dict={model.x:gray_image, model.p_keep_hidden: 1.0, model.batch_size: 1})

                    predicted_actions = sess.run(self.model.prediction, { self.X: gray_image, self.conv_keep_rate: 1.0, self.dense_keep_rate: 1.0 })[0]

                    #predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions[0]))
                    predicted_action = np.argmax(predicted_actions)
                    
                    left_action = 0.0
                    right_action = 0.0
                    throttle_action = 0.0


                    if predicted_action == 1:
                        left_action = 1.0
                    elif predicted_action == 2:
                        right_action = 1.0

                    if game.mSpeed < 50:
                        throttle_action = 0.3

                    controller.control_car(throttle_action, 0.0, left_action, right_action)
                    #controller.control_car(predicted_actions[0], predicted_actions[1], predicted_actions[2], predicted_actions[3])
                
                    #print("Throttle:", predicted_actions[0], "Brakes:", predicted_actions[1], "left:", predicted_actions[2], 'right', predicted_actions[3])

                    #print(predicted_actions)

                    #plt.matshow(np.reshape(gray_image[0],(72,128)), cmap=plt.cm.gray)
                    #plt.show()
                    #break
                    time.sleep(0.4)
                elif game_running:
                    controller.control_car(0.0, 0.0, 0.0, 0.0)
                    print('Paused')
                    game_running = False