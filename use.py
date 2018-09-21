import tensorflow as tf
from tensorflow import keras
import virtual_xbox_control as vc
import carseour as pcars
import time
from common import countdown
import numpy as np

class Use:
    def __init__(self, model_checkpoint_file_path):
        self.model_checkpoint_file_path = model_checkpoint_file_path

    def predict(self):

        model = keras.models.load_model(self.model_checkpoint_file_path)
        controller = vc.virtual_xbox_controller()

        countdown.begin_from(3)

        game = pcars.live() #game state
        position = game.mParticipantInfo[0].mWorldPosition
        angle = game.mOrientation
        velocity = game.mLocalVelocity

        is_game_playing = False

        while True:
            if game.mGameState == 2:
                is_game_playing = True

                feature = np.array([position[0], position[1], position[2],
                                    angle[0], angle[1], angle[2],
                                    velocity[0], velocity[1], velocity[2]])

                throttle_prediction = model.predict(feature)

                controller.control_car_throttle_only(throttle_prediction)

                time.sleep(0.1)
            elif is_game_playing:
                controller.control_car_throttle_only(0.0)
                print('Paused')
                #stops pause print mulitple times
                is_game_playing = False 

