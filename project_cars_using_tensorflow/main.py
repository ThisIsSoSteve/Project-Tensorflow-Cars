import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

from modes import Mode
import train
import use
from data import record
from data import data_transform

use_mode = Mode.Use

checkpoint_save_path = 'F:/Project_Cars_Data/Checkpoints/project_tensorflow_car_model.ckpt'
raw_data_save_path = 'F:/Project_Cars_Data/Raw'
training_data_save_path = 'F:/Project_Cars_Data/Training'

if use_mode == Mode.Train:
    #number, epochs, learning_rate
    train.train_model(1000, 1300, 0.001, training_data_save_path, checkpoint_save_path)

if use_mode == Mode.Use:
    use.use_model(checkpoint_save_path)

if use_mode == Mode.Record:
    #capture_rate
    record.Start(0.1, raw_data_save_path)

if use_mode == Mode.Create_Training_Data:
    data_transform.raw_to_training_data(raw_data_save_path, training_data_save_path)