import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

from modes import Mode
import train
import use
from data import record
from data import data_transform
from data import create_hdf5_file

use_mode = Mode.Use 

checkpoint_save_path = 'F:/Project_Cars_Data/Checkpoints'
checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/project_tensorflow_car_model_0.744434.ckpt-2850'
#checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/backup/80/project_tensorflow_car_model_0.800262.ckpt-1280'
raw_data_save_path = 'F:/Project_Cars_Data/Raw'
training_data_save_path = 'F:/Project_Cars_Data/Training'

if use_mode == Mode.Train:
    #number, epochs, learning_rate
    train.train_model(100000, 20000, 0.001, training_data_save_path, checkpoint_save_path, '')#21500
    #train.train_model_with_npy_file(100000, 1300, 0.001, training_data_save_path, checkpoint_save_path, '')

if use_mode == Mode.Restore_and_Train:
    #number, epochs, learning_rate
    train.train_model(100000, 20000, 0.001, training_data_save_path, checkpoint_save_path, checkpoint_use_path)
    #train.train_model_with_npy_file(100000, 1300, 0.001, training_data_save_path, checkpoint_save_path, checkpoint_use_path)

if use_mode == Mode.Use:
    use.use_model(checkpoint_use_path)

if use_mode == Mode.Record:
    #capture_rate
    record.Start(0.1, raw_data_save_path)

if use_mode == Mode.Create_Training_Data:
    #data_transform.raw_to_training_data(raw_data_save_path, training_data_save_path)
    #data_transform.raw_to_HDF5(raw_data_save_path, training_data_save_path)
    #data_transform.convert_raw_to_file(raw_data_save_path, training_data_save_path, True)
    create_hdf5_file.convert_raw_to_file(raw_data_save_path, training_data_save_path, True)
