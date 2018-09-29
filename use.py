import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force cpu use gpu cause the predict to crash

import time
import numpy as np
import carseour as pcars
from tensorflow import keras
import virtual_xbox_control as vc
from common import countdown
from data_control_no_images.read import Read

#look into this to get keras allocate memory correctly when game is running
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

class Use:
    def __init__(self, model_checkpoint_file_path, training_data_save_path):
        self.model_checkpoint_file_path = model_checkpoint_file_path
        self.training_data_save_path = training_data_save_path

    def test_predict(self):
        get_data = Read(True)
        mean, std = get_data.load_mean_and_std(self.training_data_save_path)

        model = keras.models.load_model(self.model_checkpoint_file_path)
        feature = np.array([[1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0]])

        feature = (feature - mean) / std

        print('input shape: {}'.format(feature.shape))
        throttle_prediction = model.predict(feature)
        print(throttle_prediction)


    def predict(self):

        model = keras.models.load_model(self.model_checkpoint_file_path)
        controller = vc.virtual_xbox_controller()

        countdown.begin_from(3)

        game = pcars.live() #game state
        position = game.mParticipantInfo[0].mWorldPosition
        angle = game.mOrientation
        velocity = game.mLocalVelocity

        is_game_playing = False

        get_data = Read(True)

        mean, std = get_data.load_mean_and_std(self.training_data_save_path)

        while True:
            if game.mGameState == 2:
                is_game_playing = True

                # feature = np.array([[position[0], position[1], position[2],
                #                      angle[0], angle[1], angle[2],
                #                      velocity[0], velocity[1], velocity[2]]])
                feature = np.array([[position[0], position[1], position[2],
                                      angle[0], angle[1], angle[2]]])

                #print('input: {}'.format(feature))

                feature = (feature - mean) / std

                throttle_prediction = model.predict(feature)[0]

                #print('prediction: {}'.format(throttle_prediction))

                controller.control_car_throttle_only(throttle_prediction[0])

                time.sleep(0.1)
            elif is_game_playing:
                controller.control_car_throttle_only(0.0)
                print('Paused')
                #stops multiple 'Paused' prints
                is_game_playing = False
