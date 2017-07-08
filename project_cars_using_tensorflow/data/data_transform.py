import numpy as np
import os
import cv2
import glob
import pickle
from tqdm import tqdm
from tables import *
from random import shuffle
from math import ceil

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

    current = 0

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
        gray_image = cv2.resize(gray_image, (128, 72), interpolation=cv2.INTER_CUBIC) #keep 16:9 ratio (width, height)
        #gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0
        gray_image = gray_image.reshape(72, 128, 1)

        label = np.float16([throttle / 255, brakes / 255, steering_left / 32768, steering_right / 32767]) #throttle, brakes, left, right

        training_data_array.append([gray_image, label])
        #training_data_array.append(mirror_data(gray_image, label))
        if current > 1000:
            break
        current += 1

    print('total data records', len(training_data_array)) 

    np.random.shuffle(training_data_array)

    #Split validation set from training data
    percent_of_test_data = int((len(training_data_array) / 100) * 20) #20%
    validation_data_array = np.array(training_data_array[0:percent_of_test_data])

    training_data_array = np.array(training_data_array[percent_of_test_data:])


    #PyTable Setup for training data
    path_training = training_save_path + '/training.h5'

    hdf5_file = open_file(path_training, mode = "w")

    img_dtype = UInt8Atom()
    data_shape = (0, 72, 128, 1)

    training_images_storage = hdf5_file.create_earray(hdf5_file.root, 'training_images', img_dtype, shape=data_shape)
    validation_images_storage = hdf5_file.create_earray(hdf5_file.root, 'validation_images', img_dtype, shape=data_shape)

    training_labels_storage = hdf5_file.create_earray(hdf5_file.root, 'training_labels', Float16Atom(), shape=(0,4))
    validation_labels_storage = hdf5_file.create_earray(hdf5_file.root, 'validation_labels', Float16Atom(), shape=(0,4))

    for data in tqdm(training_data_array):
        training_images_storage.append(data[0][None])
        training_labels_storage.append(data[1][None])

    for data in tqdm(validation_data_array):
        validation_images_storage.append(data[0][None])
        validation_labels_storage.append(data[1][None])

    hdf5_file.close()


def read_test():
    batch_size = 100
    path_training = 'F:/Project_Cars_Data/Training/training.h5'
    ##table = h5file.root.training.data
    ##labels = [x['label'] for x in table.iterrows()]

    #my_array = np.array([1, 2], dtype=np.int)

    hdf5_file = open_file(path_training, mode='r')
    #read = f.root.training.data
    #images, labels = [x[:] for x in read.iterrows()][3]

    data_num = hdf5_file.root.training_images.shape[0]
    print(data_num)
    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    shuffle(batches_list)

    #print(len(batches_list))

    for number, index in enumerate(batches_list):
        
        batch_starting_index = index * batch_size
        batch_ending_index = min([(index + 1) * batch_size, data_num]) 
        print(batch_starting_index, batch_ending_index)

        images = hdf5_file.root.training_images[batch_starting_index:batch_ending_index]
        labels = hdf5_file.root.training_labels[batch_starting_index:batch_ending_index]

        print(labels[0].shape)
        print(labels[0])

        print(images[0].shape)
        img = np.array(images[0])
        cv2.imshow("image", img);
        cv2.waitKey();
    
    #http://www.pytables.org/usersguide/libref/structured_storage.html
    #https://stackoverflow.com/questions/21039772/pytables-read-random-subset
    #print(images)
    #print(labels)

    ###check image
    ##img = np.array(images[0] * 255, dtype = np.uint8)
    ##cv2.imshow("image", img);
    ##cv2.waitKey();

    hdf5_file.flush()
    hdf5_file.close()
    print('Complete')

#read_test()