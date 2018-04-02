import numpy as np
import os
import cv2
import glob
import pickle
from tqdm import tqdm
from tables import *
from random import shuffle
from math import ceil
from data import data_transform

image_width = 640
image_height = 360


def convert_raw_to_file(raw_save_path, training_save_path, filename, shuffle):
    print('Starting')
     #Setup
    path_training = training_save_path + '/' + filename +'.h5' 
    if os.path.exists(path_training):
        os.remove(path_training)

    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)

    training_data_array = data_transform.raw_to_array(raw_save_path, image_height, image_width)

    if shuffle:
        np.random.shuffle(training_data_array)

    #Split validation set from training data
    percent_of_test_data = int((len(training_data_array) / 100) * 20) #20%
    validation_data_array = np.array(training_data_array[0:percent_of_test_data])

    training_data_array = np.array(training_data_array[percent_of_test_data:])

    raw_to_HDF5(path_training, training_data_array, validation_data_array)
        
    print('Complete')

def raw_to_HDF5(path_training, training_data_array, validation_data_array):

    print('creating HDF5 file')

    #PyTable Setup for training data
    #path_training = training_save_path + '/training.h5'

    hdf5_file = open_file(path_training, mode = "w")

    img_dtype = Float16Atom()
    data_shape = (0, image_height, image_width, 1)
    output_shape = (0,2)

    training_images_storage = hdf5_file.create_earray(hdf5_file.root, 'training_images', img_dtype, shape=data_shape, expectedrows=len(training_data_array))
    validation_images_storage = hdf5_file.create_earray(hdf5_file.root, 'validation_images', img_dtype, shape=data_shape, expectedrows=len(validation_data_array))

    training_labels_storage = hdf5_file.create_earray(hdf5_file.root, 'training_labels', Float16Atom(), shape=output_shape, expectedrows=len(training_data_array))
    validation_labels_storage = hdf5_file.create_earray(hdf5_file.root, 'validation_labels', Float16Atom(), shape=output_shape, expectedrows=len(validation_data_array))

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
        cv2.imshow("image", img)
        cv2.waitKey()
    
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