import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import tensorflow as tf
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from random import randint

#data/Project_Cars_2017-04-30_10-39-05

path = 'data/Project_Cars_2017-06-18_12-58-05/'

def mirror_data(traning_data_to_mirror):

    data = traning_data_to_mirror

    img = data[0]
    label = data[1]

    img = np.fliplr(img)

    choices = np.array([label[0], label[1], label[3], np.absolute(label[2])])

    return np.array([np.float16(img / 255.0), choices, data[2] / 50.0])

def put_training_data_into_one_file():
    print('put_training_data_into_one_file - Starting')

    training_data = []

    path_training = path + 'training_full.npy'
    if os.path.exists(path_training):
        os.remove(path_training)

    #Load
    for filename in tqdm(os.listdir(path)):
        data = np.load(path + filename)

        label = data[1] 
        label = np.array([label[0], label[1], np.absolute(label[2]), label[3]]) #throttle, brakes, left, right

        training_data.append([np.float16(data[0] / 255.0), label, data[2] / 50.0]) #image, labels, speed

        #mirror
        #image = np.fliplr(data[0])
        #label = data[1]
        #label = np.array([label[0], label[1], label[3]], np.absolute(label[2]))
        #training_data.append([np.float16(data[0] / 255.0), label, data[2] / 50.0])

        #training_data.append(mirror_data(data))

    np.save(path_training, training_data)
    print('put_training_data_into_one_file - Complete')

put_training_data_into_one_file()
