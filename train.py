import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
from data_control_no_images.read import Read
from plot import Plot

class Train:

    def __init__(self, checkpoint_folder_path, data_folder_path, learning_rate,
                 number_of_epochs, batch_size):
        #self.save_model_folder_path = save_model_folder_path
        self.checkpoint_folder_path = checkpoint_folder_path
        self.data_folder_path = data_folder_path
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size


    def create(self):#TODO change to proper model
        model = keras.models.Sequential([
            keras.layers.Dense(1024, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(512, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])

        #optimizer = tf.train.RMSPropOptimizer(lr=self.learning_rate)

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=['mae'])

        return model

    def model(self, restore_checkpoint_file_path='', create_sub_folder=True):

        #creates a new sub folder for the checkpoints
        if create_sub_folder:
            save_file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            self.checkpoint_folder_path = self.checkpoint_folder_path + '/' + save_file_name

            print('new checkpoint folder path: {}'.format(self.checkpoint_folder_path))
            if not os.path.exists(self.checkpoint_folder_path):
                os.makedirs(self.checkpoint_folder_path)

        #load training data
        training_features = np.load(self.data_folder_path + '/training_features.npy')
        training_labels = np.load(self.data_folder_path + '/training_labels.npy')
        #load validation data
        validation_features = np.load(self.data_folder_path + '/validation_features.npy')
        validation_labels = np.load(self.data_folder_path + '/validation_labels.npy')


        read_data = Read()
        mean, std = read_data.load_mean_and_std(self.data_folder_path)

        training_features = (training_features - mean) / std
        validation_features = (validation_features - mean) / std

        cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_folder_path + '/cp-{epoch:04d}-{val_mean_absolute_error:.2f}.h5',
                                                      save_weights_only=False, verbose=1, period=100, monitor='val_mean_absolute_error')

        if restore_checkpoint_file_path != '':
            model = keras.models.load_model(restore_checkpoint_file_path)
        else:
            model = self.create()

        history = model.fit(training_features, training_labels, epochs=self.number_of_epochs,
                  callbacks=[cp_callback], validation_data=(validation_features, validation_labels), verbose=1, batch_size=128)

        #TODO move to plot class
        new_plots = Plot(history, self.checkpoint_folder_path)
        new_plots.save_error_plot()
        new_plots.save_accuracy_plot()

        #print(history.history.keys())
        # summarize history for accuracy
        # plt.figure(1)
        # plt.plot(history.history['mean_absolute_error'])
        # plt.plot(history.history['val_mean_absolute_error'])
        # plt.title('model mean_absolute_error')
        # plt.ylabel('mean_absolute_error')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(self.checkpoint_folder_path + '/mean_absolute_error.png', dpi=128)
        # plt.show()

        # # summarize history for loss
        # plt.figure(2)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # #plt.show()
        # plt.savefig(self.checkpoint_folder_path + '/loss.png', dpi=128)
        # plt.show()
        # plt.close()


    def evaluate_test_data(self, checkpoint_file_path):
        read_data = Read()
        mean, std = read_data.load_mean_and_std(self.data_folder_path)

        #load test data
        test_features = np.load(self.data_folder_path + '/test_features.npy')
        test_labels = np.load(self.data_folder_path + '/test_labels.npy')
        test_features = (test_features - mean) / std

        model = keras.models.load_model(checkpoint_file_path)

        loss, acc = model.evaluate(test_features, test_labels)
        print("Restored model, accuracy(mae) : {}".format(acc))
        print("Restored model, loss: {}".format(loss))
    