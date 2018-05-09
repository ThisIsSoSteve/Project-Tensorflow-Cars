import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

from modes import Mode
import train
import use
from data import record
from data import data_transform
#from data import create_hdf5_file

use_mode = Mode.Train
 

checkpoint_save_path = 'F:/Project_Cars_Data/Checkpoints'
checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/project_tensorflow_car_model_0.795365.ckpt-1200'
#checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/backup/80/project_tensorflow_car_model_0.800262.ckpt-1280'
raw_data_save_path = 'F:/Project_Cars_Data/Raw'
training_data_save_path = 'F:/Project_Cars_Data/Training'


if use_mode == Mode.Train:
    #number_of_epochs, batch_size, learning_rate
    #train.train_model(100000, 4000, 0.01, training_data_save_path, checkpoint_save_path, '')#21500
    train.train_model_with_npy_file(10000, 20000, 0.01, training_data_save_path, checkpoint_save_path, '')

#if use_mode == Mode.Restore_and_Train:
    #number, epochs, learning_rate
    #train.train_model(100000, 2000, 0.0001, training_data_save_path, checkpoint_save_path, checkpoint_use_path)
    #train.train_model_with_npy_file(100000, 1300, 0.001, training_data_save_path, checkpoint_save_path, checkpoint_use_path)

#if use_mode == Mode.Use:
    #use.use_model(checkpoint_use_path)

if use_mode == Mode.Record:
    capture_rate = 0.1
    record.Start(capture_rate, raw_data_save_path)

if use_mode == Mode.Create_Training_Data:
    data_transform.get_steering_features_labels(raw_data_save_path,training_data_save_path, 32, 18)

#if use_mode == Mode.Create_Training_Data:
    #data_transform.raw_to_training_data(raw_data_save_path, training_data_save_path)
    #data_transform.raw_to_HDF5(raw_data_save_path, training_data_save_path)
    #data_transform.convert_raw_to_file(raw_data_save_path, training_data_save_path, True)
    #create_hdf5_file.convert_raw_to_file(raw_data_save_path, training_data_save_path, 'training', False)
    #create_hdf5_file.convert_raw_to_file(raw_data_save_path, training_data_save_path, 'on_track_training', True)
