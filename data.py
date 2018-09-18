import os
import pickle
import glob
import numpy as np
from tqdm import tqdm

#ToDo: Add doc strings
#ToDo: Add print statements for ease of use
#ToDo: Refactor big methods / split into classes
#ToDo: Create tests
#ToDo: Add more error catching
#ToDo: Add all project cars and controller variables for printing

class Data:

    def read_data_raw(self, file_path):
        if file_path[-4] != '.pkl':
            raise ValueError('Can\'t open {} file, please select a .pkl file.'.format(file_path[-4]))

        with open(file_path, 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        return project_cars_state, controller_state

    def print_data_raw(self, project_cars_state, controller_state):
        car = project_cars_state.mParticipantInfo[0]
        position = car.mWorldPosition

        print('Local Space: X: {}, Y: {}, Z: {}'.format(position[0], position[1], position[2]))

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

    def create_data_for_network(self, raw_folder_path, max_number_of_records=-1):
        print('starting create_none_image_features_data')

        listing = glob.glob(raw_folder_path + '/*.pkl')
        data = []

        records_added_count = 0

        for filename in tqdm(listing):

            filename = filename.replace('\\', '/')

            with open(filename, 'rb') as file_data:
                project_cars_state = pickle.load(file_data)
                controller_state = pickle.load(file_data)

            car = project_cars_state.mParticipantInfo[0]

            #remove all record that are not on a flying lap
            if car.mCurrentLapDistance == 0.0:
                continue

            position = car.mWorldPosition
            angle = project_cars_state.mOrientation
            velocity = project_cars_state.mLocalVelocity

            throttle = controller_state['right_trigger'] #0 - 255
            #brakes = controller_state['left_trigger'] #0 - 255
            #steering = controller_state['thumb_lx'] #-32768 - 32767

            feature = np.array([position[0], position[1], position[2],
                angle[0], angle[1], angle[2],
                velocity[0], velocity[1], velocity[2]])
                
            label = np.array([throttle])

            data.append([feature, label])

            records_added_count += 1
            if records_added_count == max_number_of_records:
                break

        print('Total records created: {}'.format(len(data)))
        return data

    def save_data(self, data, folder_path, percent_validation_data=15, percent_test_data=15, labeled_data_in_separate_file=True, shuffle=True):
        #make sure old data is removed
        path_training = folder_path + '/training.npy'
        if os.path.exists(path_training):
            os.remove(path_training)

        path_training_features = folder_path + '/training_features.npy'
        if os.path.exists(path_training_features):
            os.remove(path_training_features)

        path_training_labels = folder_path + '/training_labels.npy'
        if os.path.exists(path_training_labels):
            os.remove(path_training_labels)

        path_validation = folder_path + '/validation.npy'
        if os.path.exists(path_validation):
            os.remove(path_validation)

        path_validation_features = folder_path + '/validation_features.npy'
        if os.path.exists(path_validation_features):
            os.remove(path_validation_features)

        path_validation_labels = folder_path + '/validation_labels.npy'
        if os.path.exists(path_validation_labels):
            os.remove(path_validation_labels)

        path_test = folder_path + '/test.npy'
        if os.path.exists(path_test):
            os.remove(path_test)

        path_test_features = folder_path + '/validation_features.npy'
        if os.path.exists(path_test_features):
            os.remove(path_test_features)

        path_test_labels = folder_path + '/validation_labels.npy'
        if os.path.exists(path_test_labels):
            os.remove(path_test_labels)

        #create path if is does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if shuffle:
            np.random.shuffle(data)

        total_number_of_records = len(data)

        validation_percentage = int((total_number_of_records/ 100) * percent_validation_data)
        test_percentage = int((total_number_of_records / 100) * percent_test_data)

        data_training = np.array(data[test_percentage + validation_percentage:])
        data_validation = np.array(data[test_percentage: test_percentage + validation_percentage])
        data_test = np.array(data[0: test_percentage])

        #save data
        if labeled_data_in_separate_file:

            #create variables to store labels and features
            data_training_features = []
            data_training_labels = []

            data_validation_features = []
            data_validation_labels = []

            data_test_features = []
            data_test_labels = []

            for record in data_training:#is there a better way?
                data_training_features.append(np.array(record[0]))
                data_training_labels.append(np.array(record[1]))

            for record in data_validation:#is there a better way?
                data_validation_features.append(np.array(record[0]))
                data_validation_labels.append(np.array(record[1]))

            for record in data_training:#is there a better way?
                data_test_features.append(np.array(record[0]))
                data_test_labels.append(np.array(record[1]))

            np.save(path_training_features, data_training_features)
            np.save(path_training_labels, data_training_labels)

            np.save(path_validation_features, data_validation_features)
            np.save(path_validation_labels, data_validation_labels)

            np.save(path_test_features, data_test_features)
            np.save(path_test_labels, data_test_labels)
        else:
            np.save(path_training, data_training)
            np.save(path_validation, data_validation)
            np.save(path_test, data_test)

        print('Completed: Training examples: {}, Validation examples: {}, Test examples: {}'.format(len(data_training), len(data_validation), len(data_test)))

    def sample_data_for_network(self, file_path, number_of_random_sample=10):
        records = []
        if os.path.exists(file_path):
            records = np.load(file_path)
            return np.random.shuffle(records)[:number_of_random_sample]

        return records

    # def record_data_raw(self):
    #     pass
