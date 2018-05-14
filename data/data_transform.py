import numpy as np
import os
import cv2
import glob
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def convert_raw_to_file(raw_save_path, training_save_path, shuffle):
    print('starting')
     #Setup

    path_training = training_save_path + '/training.npy' 
    if os.path.exists(path_training):
        os.remove(path_training)

    path_training_test = training_save_path + '/training_validation.npy'
    if os.path.exists(path_training_test):
        os.remove(path_training_test)

    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)

    training_data_array = raw_to_array(raw_save_path, 128, 72)

    if shuffle:
        np.random.shuffle(training_data_array)

    #Split validation set from training data
    percent_of_test_data = int((len(training_data_array) / 100) * 20) #20%
    validation_data_array = np.array(training_data_array[0:percent_of_test_data])

    training_data_array = np.array(training_data_array[percent_of_test_data:])

    np.save(path_training, training_data_array)
    np.save(path_training_test, validation_data_array)

    print('Complete')

def mirror_data(image, label):

    image = np.fliplr(image)

    #choices = np.array([label[0], label[1], label[3], label[2]])

    #cv2.imshow("image", image);
    #cv2.waitKey();
    
    #return np.array([image, choices])
    return np.array([image, label])


def raw_to_array(raw_save_path, image_height, image_width):
    print('getting raw data')

    listing = glob.glob(raw_save_path + '/*.png')
    training_data_array = []

    #current = 0

    for filename in tqdm(listing):

        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')

        #Get labels
        project_cars_state = None
        controller_state = None

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        #convert image
        gray_image = cv2.imread(filename + '-image.png', cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR)#cv2.IMREAD_GRAYSCALE
        gray_image = cv2.resize(gray_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC) #keep 16:9 ratio (width, height)
        gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0
        gray_image = gray_image.reshape(image_height, image_width, 1)

        #label = get_throttle_brakes_steering_label(controller_state)
        label = get_is_car_on_track_label(project_cars_state)

        training_data_array.append([gray_image, label])
        #training_data_array.append(mirror_data(gray_image, label))
        #if current > 1000:
        #    break
        #current += 1

   
    print('total data records', len(training_data_array)) 
    return training_data_array

def get_throttle_brakes_steering_label(controller_state):
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

    return np.float16([throttle / 255, brakes / 255, steering_left / 32768, steering_right / 32767]) #throttle, brakes, left, right

def get_is_car_on_track_label(project_cars_state):
    if project_cars_state.mTerrain[0] > 4 or project_cars_state.mTerrain[1] > 4 or project_cars_state.mTerrain[2] > 4 or project_cars_state.mTerrain[3] > 4:
        return np.float16([1, 0])
    else:
        return np.float16([0, 1])

    #print('[{}][{}]'.format(game.mTerrain[0],game.mTerrain[1]))
    #print('[{}][{}]'.format(game.mTerrain[2],game.mTerrain[3]))


def get_steering_features_labels(raw_save_path, path_training, image_height, image_width):

    listing = glob.glob(raw_save_path + '/*.png')
    training_data_array = []
    test_data_array = []

    buffer = 400
    limit = 10000
    test_set_limit = limit * 0.3
    currentcount = 0 

    for filename in tqdm(listing):

        currentcount += 1

        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')
        #print(filename)

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        #only do Watkins Glen International track data
        current_track = str(project_cars_state.mTrackLocation).replace("'","").replace("b","")

        if(current_track != "Watkins Glen International"):
            continue
      

        label = np.zeros([3])

        current_steering_state = controller_state['thumb_lx']   

        #print(current_steering_state) 
       
        if current_steering_state > 0:
            label = np.array([0.0, 0.0, 1.0])#right
            #print("right")

        else:
            label = np.array([0.0, 1.0, 0.0])#left
            #print("left")

        if current_steering_state < buffer and current_steering_state > -buffer:
            label = np.array([1.0, 0.0, 0.0])#no input
            print("no input")

        gray_image = cv2.imread(filename + '-image.png', cv2.IMREAD_GRAYSCALE)

        # cv2.imshow("image", gray_image)
        # cv2.waitKey()

        gray_image = cv2.resize(gray_image, (image_height, image_width))
        gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0
        #gray_image = gray_image.reshape(image_height, image_width, 1)

        # pic = np.uint8(gray_image * 255.0)
        # plt.matshow(pic, cmap=plt.cm.gray)
        # plt.show()

        

        #previous_steering_state = current_steering_state
        if currentcount >= limit:
            test_data_array.append([gray_image, label])
        else:
            training_data_array.append([gray_image, label])

        if currentcount >= limit + test_set_limit:
            break

    np.save(path_training + '/training.npy' , training_data_array)
    np.save(path_training + '/training_validation.npy' , test_data_array)