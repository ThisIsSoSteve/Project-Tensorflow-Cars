import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

from modes import Mode
import train

from record_raw_data import record
from data_control_no_images import create
from train import Train
from use import Use
#from data import create_hdf5_file

use_mode = Mode.Create_Training_Data

checkpoint_save_path = 'F:/Project_Cars_Data/Checkpoints'
checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/project_tensorflow_car_model_0.595063.ckpt-600'
#checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/backup/80/project_tensorflow_car_model_0.800262.ckpt-1280'
raw_data_save_path = 'F:/Project_Cars_Data/Watkins Glen International - Short Circuit'
training_data_save_path = 'F:/Project_Cars_Data/Training_none_image'


if use_mode == Mode.Train:
    #number_of_epochs, batch_size, learning_rate
    training = Train(checkpoint_save_path, training_data_save_path, 0.001, 1000, 128)
    training.model()

#if use_mode == Mode.Restore_and_Train:
    #number, epochs, learning_rate
    #training = Train(image_height, image_width, label_size, 0.01)
    #training.train_model_with_npy_file(20000, 3000, training_data_save_path, checkpoint_save_path, checkpoint_use_path)

if use_mode == Mode.Use:
    using_model = Use(checkpoint_use_path)
    using_model.predict()
    #use.use_model(checkpoint_use_path)

if use_mode == Mode.Record:
    capture_rate = 0.1
    record.Start(capture_rate, raw_data_save_path)

if use_mode == Mode.Create_Training_Data:
    #data_transform.get_steering_features_labels(raw_data_save_path,training_data_save_path, image_height, image_width)
    #data_transform.convert_raw_to_file(raw_data_save_path, training_data_save_path, True)
    data = create.Create(raw_data_save_path, training_data_save_path, 100)
    data.save_data(data.network_data())