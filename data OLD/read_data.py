import pickle
import glob
import numpy as np

class read_data:

    def read_data_file(self, filename):
        project_cars_state = None
        controller_state = None

        with open(filename, 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        car = project_cars_state.mParticipantInfo[0]
        position = car.mWorldPosition

        #Local Space  X  Y  Z
        print('X: {}, Y: {}, Z: {}'.format(position[0], position[1], position[2]))

        print('Lap Distance: {} / {}'.format(car.mCurrentLapDistance, project_cars_state.mTrackLength))
        
        angle = project_cars_state.mOrientation
        print('Angle: X: {} Y: {} Z: {}'.format(angle[0], angle[1], angle[2]))

        local_velocity = project_cars_state.mLocalVelocity
        print('Local Velocity: X: {}, Y: {}, Z: {}'.format(local_velocity[0], local_velocity[1], local_velocity[2]))

        local_acceleration = project_cars_state.mLocalAcceleration
        print('Local Acceleration: X: {}, Y: {}, Z: {}'.format(local_acceleration[0], local_acceleration[1], local_acceleration[2]))

        angular_velocity = project_cars_state.mAngularVelocity
        print('Angular_Velocity: X: {}, Y: {}, Z: {}'.format(angular_velocity[0], angular_velocity[1], angular_velocity[2]))

        throttle = controller_state['right_trigger'] #0 - 255
        brakes = controller_state['left_trigger'] #0 - 255
        steering = controller_state['thumb_lx'] #-32768 - 32767

        print('Throttle: {}, Brakes: {}, Steering, {}'.format(throttle, brakes, steering))


if __name__ == "__main__":
    file_name = '2017-06-24_20-58-18-499404-data.pkl'
    file_path = 'F:/Project_Cars_Data/Watkins Glen International - Short Circuit/'
    test = read_data()

    listing = glob.glob(file_path + '/*.pkl')

    for data in listing:
        print(data)
        test.read_data_file(data)
        input("Press Enter to continue...")



    #     def normalize(train, test):
    # mean, std = train.mean(), test.std()
    # train = (train - mean) / std
    # test = (test - mean) / std
    # return train, test