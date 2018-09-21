import tensorflow as tf
import numpy as np
from tensorflow import keras
from data_control_no_images import read

class Train:

    def __init__(self, save_model_folder_path, checkpoint_folder_path, data_folder_path, learning_rate, number_of_epochs, batch_size):
        self.save_model_folder_path = save_model_folder_path
        self.checkpoint_folder_path = checkpoint_folder_path
        self.data_folder_path = data_folder_path
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size


    def create(self):#TODO change to proper model
        model = keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        model.compile(loss='mse', optimizer = optimizer, metric=['mae'])

        return model

    def model(self):
        #TODO refactor
        #load training data
        training_features = np.load(self.data_folder_path + '/training_features.npy')
        training_labels = np.load(self.data_folder_path + '/training_labels.npy')
        #load validation data
        validation_features = np.load(self.data_folder_path + '/validation_features.npy')
        validation_labels = np.load(self.data_folder_path + '/validation_labels.npy')
        #load test data
        test_features = np.load(self.data_folder_path + '/test_features.npy')
        test_labels = np.load(self.data_folder_path + '/test_labels.npy')


        #normilize data
        mean = np.mean(training_features, axis=0)
        std = np.std(training_features, axis=0)

        training_features = (training_features - mean) / std
        validation_features = (validation_features - mean) / std
        test_features = (test_features - mean) / std

        cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_folder_path, 
                                                    save_weights_only=False,
                                                    verbose=1, period=10)

        model = self.create()

        model.fit(training_features, training_labels, epochs = self.number_of_epochs,
                  callbacks = [cp_callback], validation_data = (validation_features, validation_labels), verbose=1, batch_size=128)
