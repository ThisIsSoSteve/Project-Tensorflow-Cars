import time
from common import countdown
import numpy as np

import tensorflow as tf
import model
from common import grabber
import ctypes
import cv2
import virtual_xbox_control as vc
import carseour

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

def use_model(checkpoint_save_path):
    controller = vc.virtual_xbox_controller()

    print('Get Project Cars in focus!')
    countdown.begin_from(3)
    handle = ctypes.windll.user32.GetForegroundWindow()
    grabberObject = grabber.Grabber(window=handle)

    prediction = model.myModel(model.x, model.p_keep_hidden, model.p_is_training)# model.z
    saver = tf.train.Saver()

    game_running = False
    #config = tf.ConfigProto()
   # config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session() as sess:#config=config
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_save_path)
        pic = None
        game = carseour.live()

        while True:
            
            if game.mGameState == 2:
                game_running = True
                #gameSpeed = np.array([game.mSpeed])
                #gameSpeed = np.reshape(gameSpeed, (1, 1))

                pic = grabberObject.grab(pic)#grab the screen 
                gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.resize(gray_image, (72, 128))
                gray_image = np.reshape(gray_image,(1,72,128,1)) 
                gray_image = np.float16(gray_image / 255.0)
                #gray_image[gray_image == 0] = -1.0

                #predicted_actions = sess.run(prediction, feed_dict={model.x:gray_image, model.z:gameSpeed, model.p_keep_hidden: 1.0})
                predicted_actions = sess.run(prediction, feed_dict={model.x:gray_image, model.p_keep_hidden: 1.0, model.p_is_training: False})

                predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions[0]))

                controller.control_car(predicted_actions[0], predicted_actions[1], predicted_actions[2], predicted_actions[3])
            
                #print("Throttle:", predicted_actions[0], "Brakes:", predicted_actions[1], "left:", predicted_actions[2], 'right', predicted_actions[3])

                #print(predicted_actions)

                #plt.matshow(np.reshape(gray_image[0],(72,128)), cmap=plt.cm.gray)
                #plt.show()
                #break
                time.sleep(0.01)
            elif game_running:
                controller.control_car(0, 0, 0, 0)
                print('Paused')
                game_running = False;
