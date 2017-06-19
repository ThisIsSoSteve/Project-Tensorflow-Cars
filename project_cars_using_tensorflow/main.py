import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

from modes import Mode
import train
import use

use_mode = Mode.Train

checkpoint_save_path = 'E:/repos/Project-Tensorflow-Cars/project_cars_using_tensorflow/model/project_tensorflow_car_model.ckpt'

if use_mode == Mode.Train:
    train.train_model(2000, 128, 0.001, checkpoint_save_path)

if use_mode == Mode.Use:
    use.use_model(checkpoint_save_path)
