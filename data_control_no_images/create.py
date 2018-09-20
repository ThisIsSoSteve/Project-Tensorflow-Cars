import os
import pickle
import glob
import numpy as np
from tqdm import tqdm


class Create:
    """
    Reads, transforms and saves the data in the format your network will use.

    Keyword arguments:
    raw_data_folder_path -- the Folder Path where all the raw data is saved.

    save_data_folder_path -- the Folder Path where the newly created data is stored.

    max_number_of_records -- limit the maximum number of record if -1 then all record will
    be used. (default -1)

    percent_validation_data -- the percent of the data used to create the validation set.
    (default 15)

    percent_test_data -- the percent of the data used to create the test set. (default 15)

    labeled_data_in_separate_file -- if True splits the features and labels into separate files,
    otherwise the labels an features are saved into one file. (default True)

    shuffle -- if True the data is shuffled before splitting the data out into validation
    and test sets (default True)
    """

    def __init__(self, raw_data_folder_path, save_data_folder_path, max_number_of_records=-1,
                 percent_validation_data=15, percent_test_data=15,
                 labeled_data_in_separate_file=True, shuffle=True):

        self.raw_data_folder_path = raw_data_folder_path
        self.save_data_folder_path = save_data_folder_path
        self.max_number_of_records = max_number_of_records
        self.percent_validation_data = percent_validation_data
        self.percent_test_data = percent_test_data
        self.labeled_data_in_separate_file = labeled_data_in_separate_file
        self.shuffle = shuffle

        self.path_training_features = '/training_features.npy'
        self.path_training_labels = '/training_labels.npy'
        self.path_validation_features = '/validation_features.npy'
        self.path_validation_labels = '/validation_labels.npy'
        self.path_test_features = '/validation_features.npy'
        self.path_test_labels = '/validation_labels.npy'

        self.path_training = '/training.npy'
        self.path_validation = '/validation.npy'
        self.path_test = '/test.npy'

    def network_data(self):
        """
        Reads the raw data from raw_data_folder_path (uses .pkl only)
        and transforms the data ready for saving

        returns transformed data
        """
        listing = glob.glob(self.raw_data_folder_path + '/*.pkl')
        data = []

        records_added_count = 0

        for filename in tqdm(listing):

            filename = filename.replace('\\', '/')

            with open(filename, 'rb') as file_data:
                project_cars_state = pickle.load(file_data)
                controller_state = pickle.load(file_data)

            car = project_cars_state.mParticipantInfo[0]

            # remove all record that are not on a flying lap
            if car.mCurrentLapDistance == 0.0:
                continue

            position = car.mWorldPosition
            angle = project_cars_state.mOrientation
            velocity = project_cars_state.mLocalVelocity

            throttle = controller_state['right_trigger']  # 0 - 255
            # brakes = controller_state['left_trigger'] #0 - 255
            # steering = controller_state['thumb_lx'] #-32768 - 32767

            feature = np.array([position[0], position[1], position[2],
                                angle[0], angle[1], angle[2],
                                velocity[0], velocity[1], velocity[2]])

            label = np.array([throttle])

            data.append([feature, label])

            records_added_count += 1
            if records_added_count == self.max_number_of_records:
                break

        print('Total records found: {}'.format(len(data)))
        return data

    def save_data(self, data):
        """
        Save the data to the save_data_folder_path into train, validation and test sets

        Keyword arguments:

        data -- contains the transformed data from 'network_data()'

        Note this calls remove_existing_files() before saving the data
        """

        self.remove_existing_files()

        if self.shuffle:
            np.random.shuffle(data)

        total_number_of_records = len(data)

        validation_percentage = int(
            (total_number_of_records / 100) * self.percent_validation_data)
        test_percentage = int(
            (total_number_of_records / 100) * self.percent_test_data)

        data_training = np.array(
            data[test_percentage + validation_percentage:])
        data_validation = np.array(
            data[test_percentage: test_percentage + validation_percentage])
        data_test = np.array(data[0: test_percentage])

        # save data
        if self.labeled_data_in_separate_file:

            # create variables to store labels and features
            data_training_features = []
            data_training_labels = []

            data_validation_features = []
            data_validation_labels = []

            data_test_features = []
            data_test_labels = []

            for record in data_training:  # is there a better way?
                data_training_features.append(np.array(record[0]))
                data_training_labels.append(np.array(record[1]))

            for record in data_validation:  # is there a better way?
                data_validation_features.append(np.array(record[0]))
                data_validation_labels.append(np.array(record[1]))

            for record in data_training:  # is there a better way?
                data_test_features.append(np.array(record[0]))
                data_test_labels.append(np.array(record[1]))

            np.save(self.path_training_features, data_training_features)
            np.save(self.path_training_labels, data_training_labels)

            np.save(self.path_validation_features, data_validation_features)
            np.save(self.path_validation_labels, data_validation_labels)

            np.save(self.path_test_features, data_test_features)
            np.save(self.path_test_labels, data_test_labels)
        else:
            np.save(self.path_training, data_training)
            np.save(self.path_validation, data_validation)
            np.save(self.path_test, data_test)

        print('Completed: Training examples: {}, Validation examples: {}, Test examples: {}'.format(
            len(data_training), len(data_validation), len(data_test)))

    def remove_existing_files(self):
        """
        Checks and delete existing .npy files.
        Also creates the folder path if it doesn't exist
        """

        # create path if is does not exist
        if not os.path.exists(self.save_data_folder_path):
            os.makedirs(self.save_data_folder_path)
            # as the folder was just create there's no need to check if the files exist
            return

        # make sure old data is removed
        list_of_files_to_check = [self.path_training_features,
                                  self.path_training_labels,
                                  self.path_validation_features,
                                  self.path_validation_labels,
                                  self.path_test_features,
                                  self.path_test_labels,
                                  self.path_training,
                                  self.path_validation,
                                  self.path_test]

        for file in list_of_files_to_check:
            path_training = self.save_data_folder_path + file
            if os.path.exists(path_training):
                os.remove(path_training)
