import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging

from modes import Mode
import train
import use
import record

use_mode = Mode.Use

checkpoint_save_path = 'E:/Project_Cars_Data/Checkpoints/project_tensorflow_car_model.ckpt'

if use_mode == Mode.Train:
    train.train_model(2000, 210, 0.001, checkpoint_save_path)

if use_mode == Mode.Use:
    use.use_model(checkpoint_save_path)

if use_mode == Mode.Record:
    record.Start(0.1, 'E:/Project_Cars_Data')
