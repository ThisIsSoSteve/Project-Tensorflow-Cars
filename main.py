from modes import Mode
from record_raw_data import record
from data_control_no_images.create import Create
from train import Train
from use import Use

use_mode = Mode.Use

checkpoint_save_path = 'F:/Project_Cars_Data/Checkpoints'
checkpoint_use_path = 'F:/Project_Cars_Data/Checkpoints/2018-10-02_20-09-44-296817/cp-10000-0.10.h5'
raw_data_save_path = 'F:/Project_Cars_Data/1lap-fullspeed/Watkins Glen International - Short Circuit'
training_data_save_path = 'F:/Project_Cars_Data/Full_Speed_Training_none_image'


if use_mode == Mode.Train:
    #learning_rate, number_of_epochs, batch_size
    training = Train(checkpoint_save_path, training_data_save_path, 0.0001, 10000, 2000)
    training.model()
    #training.evaluate_test_data(checkpoint_use_path)

if use_mode == Mode.Restore_and_Train:
    #learning_rate, number_of_epochs, batch_size
    training = Train(checkpoint_save_path, training_data_save_path, 0.00001, 10000, 800)
    training.model(checkpoint_use_path)

if use_mode == Mode.Use:
    using_model = Use(checkpoint_use_path, training_data_save_path)
    using_model.predict()

if use_mode == Mode.Record:
    capture_rate = 0.1
    record.Start(capture_rate, raw_data_save_path)

if use_mode == Mode.Create_Training_Data:
    data = Create(raw_data_save_path, training_data_save_path, -1, 0, 0)
    data.save_data(data.network_data())
    