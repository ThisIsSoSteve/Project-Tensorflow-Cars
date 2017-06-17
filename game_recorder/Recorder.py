import carseour
import grabber
import cv2
import ctypes
from datetime import datetime
import time
import numpy as np
import os

start_up_complete = False
capture_rate = 0.2

folder_name = 'data/Project_Cars_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    start_up_complete = True

start_countdown = False
countdown_from = 10 #seconds


def countdown(count):
    while True:
        print('Countdown -', count)
        if count == 0:
            break
        count -= 1
        time.sleep(1)

print('Get Project Cars in focus!')
countdown(10)
handle = ctypes.windll.user32.GetForegroundWindow()
grabber = grabber.Grabber(window=handle)


while start_up_complete:

    game = carseour.snapshot()

    if start_countdown and game.mGameState == 2:
        countdown(countdown_from)
        start_countdown = False

    

    #when playing the game record data
    if game.mGameState == 2:
        save_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

        pic = None

        #get data
        pic = grabber.grab(pic)
        #game_state = np.array([game.mThrottle, game.mBrake, game.mSteering])

        #game.mSpeed, game.mRpm, game.mGear,game.mTerrain[0],game.mTerrain[1],game.mTerrain[2],game.mTerrain[3]

        steer_left = 0.0
        steer_right = 0.0

        steering = game.mUnfilteredSteering
        if steering < 0:
            steer_left = np.absolute(steering)
        else:
            steer_right = steering

        game_state = np.array([game.mUnfilteredThrottle, game.mUnfilteredBrake, steer_left, steer_right])
        #print(game_state)

        #save raw data
        #np.save(folder_name + '/data-save_date' + save_date + '.npy', game_state)
        gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (128,72))#16:9 ratio
        #cv2.imwrite(folder_name + '/image-' + save_date + '.png', gray_image)

        np.save(folder_name + '/data-' + save_date + '.npy', [gray_image, game_state, game.mSpeed])
        #gray_image = None
        #pic = None

        print('Save Complete -', save_date)
    else:
        start_countdown = True

    time.sleep(capture_rate)