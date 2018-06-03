import os
import tensorflow as tf
import numpy as np
import model

from math import ceil
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from model import Model
from plot import Plot

class Train:

    def __init__(self, image_height, image_width, output_size, learning_rate = 0.01):
                
        self.image_height = image_height
        self.image_width = image_width
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.output_size])
        self.conv_keep_rate = tf.placeholder(tf.float32)
        self.dense_keep_rate = tf.placeholder(tf.float32)

        self.model = Model(self.X, self.Y, self.learning_rate, self.conv_keep_rate, self.dense_keep_rate)

        self.cost_plot = Plot([], 'Step', 'Cost')
        self.accuracy_plot = Plot([], 'Step', 'Accuracy')


    def train_model_with_npy_file(self, number_of_epochs, batch_size, training_file_path, checkpoint_file_path, checkpoint_save_path):

        print('training started')

        global_step = tf.Variable(0, trainable=False)

        # #prediction = model.myModel(model.x, model.z, model.p_keep_hidden)
        # prediction = model.myModel(model.x, model.p_keep_hidden)
        # #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=model.y))
        # #cost = tf.reduce_mean(tf.square(model.y - tf.nn.sigmoid(prediction)))

        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=model.y))
        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)#epsilon =1e-04

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model.y, 1))

        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        saver = tf.train.Saver(max_to_keep=10)

        best_accuracy = 0.20

        training_data = np.load(training_file_path + '/training.npy')
        validation_data = np.load(training_file_path + '/training_validation.npy')

        number_of_training_records = len(training_data)
        number_of_validation_records = len(validation_data)

        print('number of training records', number_of_training_records)
        print('number of validation records', number_of_validation_records)

        print('finished loading data')

        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session() as sess:#config=config

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            if checkpoint_save_path != '':
                saver.restore(sess, checkpoint_save_path)
                print('Restored Model')

            validation_x = []
            validation_y = []
            #validation_z = []
            for data in validation_data:#better way?
                    validation_x.append(np.array(data[0]))
                    validation_y.append(np.array(data[1]))
                    #validation_z.append(np.array(data[2]))
                    # print(data[1])
                    #pic = np.uint8(data[0] * 255)
                    #plt.matshow(np.reshape(pic,(model.image_height, model.image_width)), cmap=plt.cm.gray)
                    # plt.show()
            #validation_x = np.array(validation_x)
            ##test_z = test_z.reshape((-1, 1))
            step = sess.run(global_step)

            while step < number_of_epochs:
                epoch_loss = 0
                starting_batch_index = 0
                np.random.shuffle(training_data)

                train_x = []
                train_y = []
                #train_z = [] #speed

                #print(len(training_data))
                for data in training_data:#better way?

                    train_x.append(np.array(data[0]))
                    train_y.append(np.array(data[1]))
                    #train_z.append(np.array(data[2])) 

                    #print(data[1])
                    #pic = np.uint8(data[0] * 255)
                    ##plt.matshow(np.reshape(pic,(model.image_height, model.image_width)), cmap=plt.cm.gray)
                    #plt.matshow(pic, cmap=plt.cm.gray)
                    #plt.show()

                
                # print(train_y[5])
                # #pic = train_x[0].reshape((-1, model.image_height, model.image_width, 1))
                # pic = np.uint8(train_x[5] * 255)
                # plt.matshow(np.reshape(pic,(model.image_height, model.image_width)), cmap=plt.cm.gray)
                # plt.show()

                # print(train_y[6])
                # pic = np.uint8(train_x[6] * 255)
                # plt.matshow(np.reshape(pic,(model.image_height, model.image_width)), cmap=plt.cm.gray)
                # plt.show()

                # print(train_y[7])
                # pic = np.uint8(train_x[7] * 255)
                # plt.matshow(np.reshape(pic,(model.image_height, model.image_width)), cmap=plt.cm.gray)
                # plt.show()



                

                while starting_batch_index < number_of_training_records:
                    start = starting_batch_index
                    end = starting_batch_index + batch_size

                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    #batch_z = np.array(train_z[start:end])
                    #batch_z = np.reshape(batch_z ,(batch_size, 1))

                    batch_x = batch_x.reshape((-1, self.image_height, self.image_width, 1))

                    # pic = np.uint8(batch_x[0] * 255.0)
                    # plt.matshow(np.reshape(pic,(model.image_height, model.image_width)), cmap=plt.cm.gray)
                    # plt.show()


                    #_, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.z: batch_z, model.p_keep_hidden: 0.8})
                    _, loss_val = sess.run(self.model.optimize, {self.X: batch_x, self.Y: batch_y, self.conv_keep_rate: 0.5, self.dense_keep_rate: 0.7})

                    epoch_loss += loss_val
                    starting_batch_index += batch_size
                
                step += 1
                sess.run(global_step.assign(step))
                print('Global Step', step, 'Loss', epoch_loss)
                

                if(step % 10 == 0):
                    current_accuracy = 0
                    current_accuracys = []
                    starting_batch_index = 0

                    while starting_batch_index < number_of_validation_records:

                        start = starting_batch_index
                        end = starting_batch_index + batch_size

                        validation_batch_x = np.array(validation_x[start:end])
                        validation_batch_y = np.array(validation_y[start:end])

                        validation_batch_x = validation_batch_x.reshape((-1, self.image_height, self.image_width, 1))

                        current_accuracy = sess.run(self.model.error, { self.X: validation_batch_x, self.Y: validation_batch_y, self.conv_keep_rate: 1.0, self.dense_keep_rate: 1.0})

                        #current_accuracys.append(accuracy.eval(feed_dict={model.x: validation_batch_x, model.y: validation_batch_y, model.p_keep_hidden: 1.0}))
                        current_accuracys.append(current_accuracy)

                        starting_batch_index += batch_size
                    current_accuracy = np.average(current_accuracys)
                    
                    print('Accuracy %g' % current_accuracy)
                    if current_accuracy > best_accuracy or step % 100 == 0:
                        best_accuracy = current_accuracy
                        saver.save(sess, checkpoint_file_path + '/project_tensorflow_car_model_' + str(current_accuracy) +'.ckpt', global_step=step)
                        print('Saved CheckPoint', str(current_accuracy) )

                    self.cost_plot.data.append(epoch_loss)
                    self.accuracy_plot.data.append(current_accuracy)

                    self.cost_plot.save_sub_plot(self.accuracy_plot, training_file_path + "/{} and {}.png".format(self.cost_plot.y_label, self.accuracy_plot.y_label))