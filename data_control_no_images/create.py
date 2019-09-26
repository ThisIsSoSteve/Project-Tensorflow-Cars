import os
import pickle
import glob
import numpy as np
from tqdm import tqdm
import random

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
        self.path_test_features = '/test_features.npy'
        self.path_test_labels = '/test_labels.npy'

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

        if self.shuffle:
            np.random.shuffle(listing)

        prev_steering_feature = 0
        prev_throttle_feature = 0

        for filename in tqdm(listing):

            filename = filename.replace('\\', '/')

            with open(filename, 'rb') as file_data:
                project_cars_state = pickle.load(file_data)
                controller_state = pickle.load(file_data)

            car = project_cars_state.mParticipantInfo[0]

            # remove all record that are not on a flying lap
            

            position = car.mWorldPosition
            #angle = project_cars_state.mOrientation
            #velocity = project_cars_state.mLocalVelocity
            #speed = project_cars_state.mSpeed


            throttle = controller_state['right_trigger'] / 255.0 # 0 - 255
            brakes = controller_state['left_trigger'] / 255.0#0 - 255
            steering = controller_state['thumb_lx'] /32767#-32768 - 32767


            # feature = np.array([position[0], position[1], position[2],
            #                     angle[0], angle[1], angle[2],
            #                     velocity[0], velocity[1], velocity[2]])

            # feature = np.array([position[0], position[1], position[2],
            #                     angle[0], angle[1], angle[2]])
            #position 2 is up and down

            # round(angle[1], 1)]
            #feature = np.array([position[0], position[1], position[2], angle[1]])
            
            #rolling_previous_features = np.append(rolling_previous_features, [feature], axis=0)
            #rolling_previous_features.append([feature])
            #rolling_previous_features = rolling_previous_features[1:]

            # if rolling_previous_features[0][0] == 0.0 and rolling_previous_features[0][3] == 0.0:
            #     continue

            #print(rolling_previous_features)

            #print(feature.shape())

            # label = np.array([throttle, brakes, steering])
            #print(project_cars_state.mSteering)
            # if project_cars_state.mSteering == 0.0:
            #     #print(steering)
            #     if random.randint(0,2) == 0:
            #         continue
            # if throttle > 0.0 and brakes > 0.0:#not many (4)
            #     print('throttle {}, brakes {}'.format(throttle, brakes))

            # speed = throttle / 255.0
            # speed = speed + ((brakes / 255.0) * -1)

            # label = np.array([speed, steering / 32767.0])

            feature = np.array([round(position[0], 2), round(position[2], 2), prev_steering_feature, prev_throttle_feature])

            throttle_label = throttle
            throttle_label = throttle_label + ((brakes) * -1)

            label = np.array([throttle_label, steering])
            #label = np.array([speed])
            #label = np.array([steering / 32768])

            prev_steering_feature = steering
            prev_throttle_feature = throttle_label

            if car.mCurrentLapDistance == 0.0:
                continue

            data.append([feature, label])

            # new_features = np.array([rolling_previous_features[0][0], rolling_previous_features[0][1], rolling_previous_features[0][2], rolling_previous_features[0][3],
            #     rolling_previous_features[1][0], rolling_previous_features[1][1], rolling_previous_features[1][2], rolling_previous_features[1][3],
            #     rolling_previous_features[2][0], rolling_previous_features[2][1], rolling_previous_features[2][2], rolling_previous_features[2][3]])

            #data.append([new_features, label])

            records_added_count += 1
            if records_added_count == self.max_number_of_records:
                break

        print('Total records found: {}'.format(len(data)))


        for data_record in data:
            print(data_record)
        
        return data

    def save_data(self, data):
        """
        Save the data to the save_data_folder_path into train, validation and test sets

        Keyword arguments:

        data -- contains the transformed data from 'network_data()'

        Note this calls remove_existing_files() before saving the data
        """
        

        self.remove_existing_files()

       

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

            throttle_count = 0
            no_throttle_count = 0

            no_steering_count = 0

            
            #np.random.shuffle(data_training)

            for record in data_training:  # is there a better way?
                #print(record[1])
                #print(record[1] / 32768)
                

                
                #temp_record = temp_record + ((record[1][1] / 255.0) * -1)
                
                balance_throttle = False
                if balance_throttle:
                    temp_record = record[1][0]
                    if temp_record > 0.0:
                        if no_throttle_count > throttle_count:
                            data_training_features.append(np.array(record[0]))
                            data_training_labels.append(record[1])
                            throttle_count += 1
                            print(temp_record)
                    else:
                        data_training_features.append(np.array(record[0]))
                        data_training_labels.append(record[1])
                        no_throttle_count += 1
                        print(temp_record)
                else:
                    data_training_features.append(np.array(record[0]))
                    data_training_labels.append(record[1])
                    #print(record[1])
                # if no_throttle_count < throttle_count:
                #     continue
                
                
                


                #if record[1] <= -0.5:
                
                


                

            # for record in data_training:  # is there a better way?
            #     # temp_record = record[1][0] / 255.0
            #     # temp_record = temp_record + ((record[1][1]  / 255.0) * -1)

            #     if record[1] >= 0.5 and throttle_count < no_throttle_count:
            #         data_training_features.append(np.array(record[0]))
            #         data_training_labels.append(np.array(record[1]))
            #         throttle_count += 1

            # for record in data_training:  # is there a better way?
            #     # temp_record = record[1][0] / 255.0
            #     # temp_record = temp_record + ((record[1][1]  / 255.0) * -1)

            #     if record[1] > -0.02 and record[1] < 0.02 and no_steering_count < no_throttle_count:
            #         data_training_features.append(np.array(record[0]))
            #         data_training_labels.append(np.array(0.0))
            #         no_steering_count += 1


            for record in data_validation:  # is there a better way?
                data_validation_features.append(np.array(record[0]))
                data_validation_labels.append(np.array(record[1]))

            for record in data_test:  # is there a better way?
                data_test_features.append(np.array(record[0]))
                data_test_labels.append(np.array(record[1]))

            #print('data {}'.format(len(data_training_features)))
            print('throttle : {} brake: {} '.format(throttle_count, no_throttle_count))


            np.save(self.save_data_folder_path + self.path_training_features, data_training_features)
            np.save(self.save_data_folder_path + self.path_training_labels, data_training_labels)

            np.save(self.save_data_folder_path + self.path_validation_features, data_validation_features)
            np.save(self.save_data_folder_path + self.path_validation_labels, data_validation_labels)

            np.save(self.save_data_folder_path + self.path_test_features, data_test_features)
            np.save(self.save_data_folder_path + self.path_test_labels, data_test_labels)

            self.save_mean_and_std()
        else:
            np.save(self.path_training, data_training)
            np.save(self.path_validation, data_validation)
            np.save(self.path_test, data_test)

        print('Completed: Training examples: {}, Validation examples: {}, Test examples: {}'.format(
            len(data_training_features), len(data_validation_features), len(data_test_features)))

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

    def save_mean_and_std(self):
         #load training data
        training_features = np.load(self.save_data_folder_path + self.path_training_features)
        #training_labels = np.load(self.save_data_folder_path + self.path_training_features)

        #normilize data
        mean = np.mean(training_features, axis=0)
        std = np.std(training_features, axis=0)

        np.save(self.save_data_folder_path + '/mean.npy', mean)
        np.save(self.save_data_folder_path + '/std.npy', std)
