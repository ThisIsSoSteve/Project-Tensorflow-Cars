import carseour as pcars
import pyxinput

from common import grabber
import cv2
import ctypes

from datetime import datetime
import time
from common import countdown

import numpy as np
import os
import pickle

#seems to be problem with pylint thinking cv2 has no Members 
import win32gui, win32ui, win32con, win32api




def Start(capture_rate, root_save_folder):
    start_up_complete = False
    #capture_rate = 0.1

    if capture_rate <= 0:
        raise ValueError('Capture rate has to higher than 0')

    if root_save_folder == '':
        raise ValueError('Please specify a root directory e.g E:/myData')

    folder_name = root_save_folder #+ '/Project_Cars_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #folder_name = 'E:/Project_Cars_Data/Project_Cars_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    start_up_complete = True

    start_countdown = False
    countdown_from = 10 #seconds

    print('Get Project Cars in focus!')
    countdown.begin_from(3)
    handle = ctypes.windll.user32.GetForegroundWindow()
    #grabber1 = grabber.Grabber(window=handle)

    project_cars_state = pcars.live()
    #game = carseour.snapshot()
    controller_state = pyxinput.rController(1)

    while start_up_complete:

        if start_countdown and project_cars_state.mGameState == 2:
            countdown.begin_from(countdown_from)
            start_countdown = False

        #when playing the game record data
        if project_cars_state.mGameState == 2:
            save_file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

            #get data
            #pic = grabber1.grab()
            #game_state = np.array([game.mThrottle, game.mBrake, game.mSteering])

            #game.mSpeed, game.mRpm, game.mGear,game.mTerrain[0],game.mTerrain[1],game.mTerrain[2],game.mTerrain[3]
            #game_state = np.array([game.mUnfilteredThrottle, game.mUnfilteredBrake, game.mUnfilteredSteering])
            #print(game_state)

            #save raw data
            #np.save(folder_name + '/data-save_date' + save_date + '.npy', game_state)
            #gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            #gray_image = cv2.resize(gray_image, (128,72))#16:9 ratio

            read = controller_state.gamepad
            controls = {'wButtons' : read.wButtons,
                        'left_trigger': read.left_trigger, 
                        'right_trigger' : read.right_trigger, 
                        'thumb_lx': read.thumb_lx, 
                        'thumb_ly': read.thumb_ly, 
                        'thumb_rx': read.thumb_rx, 
                        'thumb_ry': read.thumb_ry}


            gtawin = win32gui.FindWindow(None, 'Project CARS™')
            if not gtawin:
                raise Exception('window title not found')

            left, top, x2, y2 = win32gui.GetWindowRect(gtawin)
            #width = x2 - left
            #height = y2 - top
            width = 1920
            height = 1080

            #projects windows adjustments
            left = left+ 8
            top = top + 31




            print('w:{} h:{}'.format(width, height))


            hwin = win32gui.GetDesktopWindow()
            # width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            # height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            # left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            # top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            hwindc = win32gui.GetWindowDC(hwin)
            srcdc = win32ui.CreateDCFromHandle(hwindc)
            memdc = srcdc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(srcdc, width, height)
            memdc.SelectObject(bmp)
            memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
            bmp.SaveBitmapFile(memdc, folder_name + '/' + save_file_name + '-image.bmp')

            # pic = cv2.resize(pic, (640,360))
            # cv2.imshow("image", pic)
            # cv2.waitKey()
            
            #cv2.imwrite(folder_name + '/' + save_file_name + '-image.png', pic)
            with open(folder_name + '/' + save_file_name + '-data.pkl', 'wb') as output:
                pickle.dump(project_cars_state, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(controls, output, pickle.HIGHEST_PROTOCOL)

         
            #np.save(folder_name + '/data-' + save_date + '.npy', [gray_image, game_state, game.mSpeed])
            #gray_image = None
            #pic = None

            srcdc.DeleteDC()
            memdc.DeleteDC()
            win32gui.ReleaseDC(hwin, hwindc)
            win32gui.DeleteObject(bmp.GetHandle())


            print('Save Complete -', save_file_name)
        else:
            start_countdown = True

        time.sleep(capture_rate)


#def test_read(full_path):
#    #pyxinput.test_read()
#    with open(full_path, 'rb') as input:
#        project_cars_state = pickle.load(input)
#        controller_state = pickle.load(input)

#        print('carseour', project_cars_state.mThrottle)
#        print('controller', controller_state.gamepad)

#test_read('E:/Project_Cars_Data/Project_Cars_2017-06-22_20-58-04/2017-06-22_20-58-15-070058-data.pkl')
#https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence