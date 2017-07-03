import numpy as np
import os
import cv2
import glob
import pickle
from tqdm import tqdm
from tables import *


class training_data(IsDescription):
    image = Float16Col(shape=(72, 128))
    label = Float16Col(shape=(4))

def mirror_data(image, label):

    image = np.fliplr(image)

    choices = np.array([label[0], label[1], label[3], label[2]])

    #cv2.imshow("image", image);
    #cv2.waitKey();
    
    return np.array([image, choices])


def raw_to_HDF5(raw_save_path, training_save_path):

    print('Starting')

    #Setup
    path_training = training_save_path + '/training.npy'
    if os.path.exists(path_training):
        os.remove(path_training)

    path_training_test = training_save_path + '/training_test.npy'
    if os.path.exists(path_training_test):
        os.remove(path_training_test)

    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)

    listing = glob.glob(raw_save_path + '/*.png')

    training_data_array = []

    #current = 0

    for filename in tqdm(listing):

        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')

        training_data_record = []

        #Get labels
        label = []
        project_cars_state = None
        controller_state = None

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

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

        #convert image
        gray_image = cv2.imread(filename + '-image.png', cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR)#cv2.IMREAD_GRAYSCALE
        gray_image = cv2.resize(gray_image, (128, 72)) #16:9 ratio
        gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0

        label = np.float16([throttle / 255, brakes / 255, steering_left / 32768, steering_right / 32767]) #throttle, brakes, left, right

        training_data_array.append([gray_image, label])

        #if current > 1000:
        #    break
        #current += 1

    np.random.shuffle(training_data_array)
    percent_of_test_data = int((len(training_data_array) / 100) * 20) #20%

    test_data_array = np.array(training_data_array[0:percent_of_test_data])
    training_data_array = np.array(training_data_array[percent_of_test_data:])

    #PyTable Setup for training data
    path_training = training_save_path + '/training.h5'

    h5file = open_file(path_training, mode = "w", title = "Training Data")

    group = h5file.create_group("/", 'training', 'Training information')

    table = h5file.create_table(group, 'data', training_data, "Data")
    #print(h5file)
    training_data_pointer = table.row

    for data in tqdm(training_data_array):
        training_data_pointer['image'] = data[0] 
        training_data_pointer['label'] = data[1]
        training_data_pointer.append()

    table.flush()
    h5file.close()

    #PyTable Setup for testing data
    path_training = training_save_path + '/training_test.h5'

    h5file = open_file(path_training, mode = "w", title = "Training Data")

    group = h5file.create_group("/", 'training', 'Training information')

    table = h5file.create_table(group, 'data', training_data, "Data")

    training_data_pointer = table.row

    for data in tqdm(test_data_array):
        training_data_pointer['image'] = data[0] 
        training_data_pointer['label'] = data[1]
        training_data_pointer.append()

    table.flush()
    h5file.close()


    ##read test
    ##table = h5file.root.training.data
    ##labels = [x['label'] for x in table.iterrows()]
    #f = open_file(path_training, mode='r')
    #read = f.root.training.data
    #images = [x['image'] for x in read.iterrows()]

    ##print(images[0])

    ###check image
    ##img = np.array(images[0] * 255, dtype = np.uint8)
    ##cv2.imshow("image", img);
    ##cv2.waitKey();

    #f.flush()
    #f.close()
    #print('Complete')

    #todo read h5 in training