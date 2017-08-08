
import time
from common import countdown
import numpy as np

import tensorflow as tf
import model
from common import grabber
import ctypes
import cv2
import carseour

def use_model(checkpoint_save_path):

    game_running = False
    print('Get Project Cars in focus!')
    countdown.begin_from(10)
    handle = ctypes.windll.user32.GetForegroundWindow()
    grabberObject = grabber.Grabber(window=handle)

    prediction = model.myModel(model.x, model.p_keep_hidden)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_save_path)
        pic = None
        game = carseour.live()

        while True:
            
            if game.mGameState == 2:
                game_running = True

                pic = grabberObject.grab(pic)#grab the screen 
                gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.resize(gray_image, (128, 72), interpolation=cv2.INTER_CUBIC)
                gray_image = np.float16(gray_image / 255.0)
                gray_image = np.reshape(gray_image, (-1,72,128,1)) 
              
                predicted_actions = sess.run(prediction, feed_dict={model.x:gray_image, model.p_keep_hidden: 1.0})

                predicted_actions = sess.run(tf.nn.softmax(predicted_actions[0]))

                result = np.argmax(predicted_actions)

                #print(result)
                print('[{}][{}]'.format(game.mTerrain[0],game.mTerrain[1]))
                print('[{}][{}]'.format(game.mTerrain[2],game.mTerrain[3]))
                if result == 0:
                    print('Off Track!')
                else:
                    print('On Track')
                    

                #print(predicted_actions)
                time.sleep(0.5)
            elif game_running:
                print('Paused')
                game_running = False;
