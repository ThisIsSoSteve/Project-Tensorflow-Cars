import numpy as np
import os
import cv2
import glob
import pickle
from tqdm import tqdm
import image

def raw_to_training_data(raw_save_path, training_save_path):
    print('Starting')

    training_data = []

    path_training = training_save_path + '/training.npy'
    if os.path.exists(path_training):
        os.remove(path_training)

    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)

    listing = glob.glob(raw_save_path + '/*.png')

    for filename in tqdm(listing):
        
        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')

        training_data_record = []

        label = []
        project_cars_state = None
        controller_state = None

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        speed = project_cars_state.mSpeed

        throttle = controller_state['right_trigger'] #0 - 255
        brakes = controller_state['left_trigger'] #0 - 255
        steering = controller_state['thumb_lx'] #-32768 - 32767

        steering_left = 0
        steering_right = 0

        if steering < 0:
            steering_left = np.absolute(steering)
            steering_right = 0
        else:
            steering_right = steering
            steering_left = 0
        #print('speed:', speed, 'throttle:', throttle / 255, 'brakes', brakes / 255, 'steer left', steering_left / 32768, 'steer right', steering_right / 32767)
        #print('speed:', speed, 'throttle:', project_cars_state.mUnfilteredThrottle , 'brakes', project_cars_state.mUnfilteredBrake, 'steering', project_cars_state.mUnfilteredSteering)
        
        gray_image = cv2.imread(filename + '-image.png', cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR)#cv2.IMREAD_GRAYSCALE
        #gray_image = image.filter(gray_image)
        #gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (128,72)) #16:9 ratio
        gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0

        label = np.float16([throttle / 255, brakes / 255, steering_left / 32768, steering_right / 32767]) #throttle, brakes, left, right
        #label = np.array([project_cars_state.mUnfilteredThrottle, project_cars_state.mUnfilteredBrake, steering_left, steering_right])
        training_data.append([gray_image, label]) #,speed

    np.save(path_training, training_data)

    print('Complete')