import pickle
import os
import numpy as np

#ToDo add get all data (training, validation, test)
class Read:
    """
    Can read the raw or transformed data

    Keyword arguments:

    display_debug_info -- If True the data will be printed in the console

    """

    def __init__(self, display_debug_info=False):
        self.display_debug_info = display_debug_info

    def get(self, file_path, number_of_random_sample=10):
        """
        Read train, validation or test .npy files and selects a sample.

        Keyword arguments:
        file_path -- the full path to the file you want to read .npy

        number_of_random_sample -- select random number of records to return (default 10)

        returns array of records.

        """

        if file_path[-4] != '.npy':
            raise ValueError('Can\'t open {} file, please select a .npy file.'
                             .format(file_path[-4]))

        records = []
        if os.path.exists(file_path):
            records = np.load(file_path)
            return np.random.shuffle(records)[:number_of_random_sample]

        return records

    def get_raw(self, file_path):
        """
        Loads a .pkl which contains the games state and controller state.

        Will error if file is not a .pkl

        Keyword arguments:
        file_path -- the full path to the file you want to read .pkl

        """

        if file_path[-4] != '.pkl':
            raise ValueError('Can\'t open {} file, please select a .pkl file.'
                             .format(file_path[-4]))

        with open(file_path, 'rb') as data:
            project_cars_state = pickle.load(data)
            controller_state = pickle.load(data)

        if self.display_debug_info:
            self.print_raw(project_cars_state, controller_state)

        return project_cars_state, controller_state

    def print_raw(self, project_cars_state, controller_state):
        """
        Prints the raw data into a readable format.

        Use get(filepath)

        project_cars_state -- the recorded games values
        controller_state -- the controllers inputs
        """

        car = project_cars_state.mParticipantInfo[0]
        position = car.mWorldPosition

        print('Local Space: X: {}, Y: {}, Z: {}'.format(
            position[0], position[1], position[2]))

        print('Lap Distance: {} / {}'
              .format(car.mCurrentLapDistance, project_cars_state.mTrackLength))

        angle = project_cars_state.mOrientation
        print('Angle: X: {} Y: {} Z: {}'.format(angle[0], angle[1], angle[2]))

        local_velocity = project_cars_state.mLocalVelocity
        print('Local Velocity: X: {}, Y: {}, Z: {}'
              .format(local_velocity[0], local_velocity[1], local_velocity[2]))

        local_acceleration = project_cars_state.mLocalAcceleration
        print('Local Acceleration: X: {}, Y: {}, Z: {}'
              .format(local_acceleration[0], local_acceleration[1], local_acceleration[2]))

        angular_velocity = project_cars_state.mAngularVelocity
        print('Angular_Velocity: X: {}, Y: {}, Z: {}'
              .format(angular_velocity[0], angular_velocity[1], angular_velocity[2]))

        throttle = controller_state['right_trigger']  # 0 - 255
        brakes = controller_state['left_trigger']  # 0 - 255
        steering = controller_state['thumb_lx']  # -32768 - 32767

        print('Throttle: {}, Brakes: {}, Steering, {}'.format(
            throttle, brakes, steering))


    def load_mean_and_std(self, file_path):
        mean = np.load(file_path + '/mean.npy')
        std = np.load(file_path + '/std.npy')

        if self.display_debug_info:
            print('mean: {}'.format(mean))
            print('std: {}'.format(std))

        return mean, std