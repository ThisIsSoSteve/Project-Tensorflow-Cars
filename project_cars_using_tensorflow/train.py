import os
import tensorflow as tf
import numpy as np
import model
from tables import *
from math import ceil
import time

def train_model(number_of_epochs, batch_size, learning_rate, training_file_path, checkpoint_file_path, checkpoint_save_path):

    path_training = training_file_path + '/training.h5'

    global_step = tf.Variable(0, trainable=False)

    #prediction = model.myModel(model.x, model.z, model.p_keep_hidden)
    prediction = model.myModel(model.x, model.p_keep_hidden, model.p_is_training)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=model.y))
    #cost = tf.reduce_mean(tf.square(model.y - tf.nn.sigmoid(prediction)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)#epsilon =1e-04

    correct = tf.equal(tf.round(tf.nn.sigmoid(prediction) * 10.0), tf.round(model.y) * 10.0)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver(max_to_keep=10)

    best_accuracy = 0.3

    hdf5_file = open_file(path_training, mode='r', driver="H5FD_CORE")

    data_num = hdf5_file.root.training_images.shape[0]
    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    
    validation_data_num = hdf5_file.root.validation_images.shape[0]
    # create validation list of batches
    validation_batches_list = list(range(int(ceil(float(validation_data_num) / batch_size))))

    print('training number', data_num, 'validation number', validation_data_num)

    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session() as sess:#config=config

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if checkpoint_save_path != '':
            saver.restore(sess, checkpoint_save_path)
        step = sess.run(global_step)

        while step < number_of_epochs:
            epoch_loss = 0
            np.random.shuffle(batches_list)

            for number, index in enumerate(batches_list):
                batch_starting_index = index * batch_size
                batch_ending_index = min([(index + 1) * batch_size, data_num]) 

                batch_x = hdf5_file.root.training_images[batch_starting_index:batch_ending_index]
                batch_y = hdf5_file.root.training_labels[batch_starting_index:batch_ending_index]

                _, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.p_keep_hidden: 0.9, model.p_is_training: True})

                epoch_loss += loss_val

            step += 1
            sess.run(global_step.assign(step))
            print('Global Step', step, 'Loss', epoch_loss)

            if(step % 10 == 0):
                current_accuracy = 0
                batch_starting_index = 0
                current_accuracys = []
                for number, index in enumerate(validation_batches_list):
                    batch_starting_index = index * batch_size
                    batch_ending_index = min([(index + 1) * batch_size, validation_data_num]) 
                    validation_batch_x = hdf5_file.root.validation_images[batch_starting_index:batch_ending_index]
                    validation_batch_y = hdf5_file.root.validation_labels[batch_starting_index:batch_ending_index]
                    current_accuracys.append(accuracy.eval(feed_dict={model.x: validation_batch_x, model.y: validation_batch_y, model.p_keep_hidden: 1.0, model.p_is_training: False}))
                print(current_accuracys)
                current_accuracy = np.average(current_accuracys)
               
                print('Accuracy %g' % current_accuracy)
                if current_accuracy > best_accuracy or step % 50 == 0:
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    saver.save(sess, checkpoint_file_path + '/project_tensorflow_car_model_' + str(current_accuracy) +'.ckpt', global_step=step)
                    print('Saved CheckPoint', str(current_accuracy) )

    hdf5_file.flush()
    hdf5_file.close()


def train_model_with_npy_file(number_of_epochs, batch_size, learning_rate, training_file_path, checkpoint_file_path, checkpoint_save_path):

    print('starting')
    path_training = training_file_path + '/training.npy'

    path_tvalidation = training_file_path + '/training_validation.npy'

    global_step = tf.Variable(0, trainable=False)

    #prediction = model.myModel(model.x, model.z, model.p_keep_hidden)
    prediction = model.myModel(model.x, model.p_keep_hidden, model.p_is_training)
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=model.y))
    cost = tf.reduce_mean(tf.square(model.y - tf.nn.sigmoid(prediction)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)#epsilon =1e-04

    correct = tf.equal(tf.round(tf.nn.sigmoid(prediction) * 10.0), tf.round(model.y) * 10.0)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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

        validation_x = []
        validation_y = []
        #validation_z = []
        for data in validation_data:#better way?
                validation_x.append(np.array(data[0]))
                validation_y.append(np.array(data[1]))
                #validation_z.append(np.array(data[2]))

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
            for data in training_data:#better way?

                train_x.append(np.array(data[0]))
                train_y.append(np.array(data[1]))
                #train_z.append(np.array(data[2])) 

            while starting_batch_index < number_of_training_records:
                start = starting_batch_index
                end = starting_batch_index + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                #batch_z = np.array(train_z[start:end])
                #batch_z = np.reshape(batch_z ,(batch_size, 1))

                batch_x = batch_x.reshape((-1, model.image_height, model.image_width, 1))

                #_, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.z: batch_z, model.p_keep_hidden: 0.8})
                _, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.p_keep_hidden: 0.8, model.p_is_training: True})

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

                    validation_batch_x = validation_batch_x.reshape((-1, model.image_height, model.image_width, 1))

                    current_accuracys.append(accuracy.eval(feed_dict={model.x: validation_batch_x, model.y: validation_batch_y, model.p_keep_hidden: 1.0, model.p_is_training: False}))
                    starting_batch_index += batch_size
                current_accuracy = np.average(current_accuracys)
               
                print('Accuracy %g' % current_accuracy)
                if current_accuracy > best_accuracy or step % 100 == 0:
                    best_accuracy = current_accuracy
                    saver.save(sess, checkpoint_file_path + '/project_tensorflow_car_model_' + str(current_accuracy) +'.ckpt', global_step=step)
                    print('Saved CheckPoint', str(current_accuracy) )