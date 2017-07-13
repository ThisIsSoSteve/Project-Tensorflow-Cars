import os
import tensorflow as tf
import numpy as np
import model
from tables import *
from math import ceil

#def init_batches(training_file_path, batch_size):
#     #load data
#    print('Loading Training Data')
#    training_data = np.load(training_file_path + '/training.npy')
#    print('Loaded Training Data')
#    np.random.seed(0)
#    np.random.shuffle(training_data)
#    percent_of_test_data = int((len(training_data) / 100) * 2) 

#   # print(len(training_data))
#    test_data = np.array(training_data[0:percent_of_test_data])
#    training_data = np.array(training_data[percent_of_test_data:])

#    #print(len(training_data))

#    return training_data, test_data

#def next_batch(current_batch_index, batch_size):
#    start = current_batch_index
#    end = current_batch_index + batch_size

#    return np.array(training_data[0][start:end]), np.array(training_data[1][start:end])


def train_model(number_of_epochs, batch_size, learning_rate, training_file_path, checkpoint_file_path, checkpoint_save_path):

    #training_data, test_data = init_batches(training_file_path, batch_size)

    path_training = training_file_path + '/training.h5'

    global_step = tf.Variable(0, trainable=False)

    #prediction = model.myModel(model.x, model.z, model.p_keep_hidden)
    prediction = model.myModel(model.x, model.p_keep_hidden, model.p_is_training)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=model.y))
   # cost = tf.reduce_mean(tf.square(model.y - tf.nn.sigmoid(prediction)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)#epsilon =1e-04

    correct = tf.equal(tf.round(tf.nn.sigmoid(prediction) * 10.0), tf.round(model.y) * 10.0)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver(max_to_keep=10)

    best_accuracy = 0.20

    hdf5_file = open_file(path_training, mode='r')#driver="H5FD_CORE"

    data_num = hdf5_file.root.training_images.shape[0]
    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    
    validation_data_num = hdf5_file.root.validation_images.shape[0]
    # create validation list of batches
    validation_batches_list = list(range(int(ceil(float(validation_data_num) / batch_size))))

    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session() as sess:#config=config

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if checkpoint_save_path != '':
            saver.restore(sess, checkpoint_save_path)

        #test_x = []
        #test_y = []
        ##test_z = []
        #for data in test_data:#better way?
        #        test_x.append(np.array(data[0]))
        #        test_y.append(np.array(data[1]))
        #        #test_z.append(np.array(data[2]))

        #test_x = np.array(test_x)
        #test_x = test_x.reshape((-1, model.image_height, model.image_width, 1))
        ##test_z = test_z.reshape((-1, 1))
        step = sess.run(global_step)

        while step < number_of_epochs:
            epoch_loss = 0
            #starting_batch_index = 0
            #np.random.shuffle(training_data)
            np.random.shuffle(batches_list)
            
            #train_x = []
            #train_y = []
            ##train_z = [] #speed
            #for data in training_data:#better way?

            #    train_x.append(np.array(data[0]))
            #    train_y.append(np.array(data[1]))
            #    #train_z.append(np.array(data[2])) 
            
            for number, index in enumerate(batches_list):
            #while starting_batch_index < len(training_data[0]):
                batch_starting_index = index * batch_size
                batch_ending_index = min([(index + 1) * batch_size, data_num]) 

                batch_x = hdf5_file.root.training_images[batch_starting_index:batch_ending_index]
                batch_y = hdf5_file.root.training_labels[batch_starting_index:batch_ending_index]

                #batch_x = np.array(train_x[start:end])
                #batch_y = np.array(train_y[start:end])
                ##batch_z = np.array(train_z[start:end])
                ##batch_z = np.reshape(batch_z ,(batch_size, 1))

                #batch_x = batch_x.reshape((-1, model.image_height, model.image_width, 1))

                #_, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.z: batch_z, model.p_keep_hidden: 0.8})
                _, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.p_keep_hidden: 0.8, model.p_is_training: True})

                epoch_loss += loss_val
                #starting_batch_index += batch_size

            
            #step = sess.run(global_step)
            step += 1
            sess.run(global_step.assign(step))
            print('Global Step', step, 'Loss', epoch_loss)
            

            #if(step % 500 == 0):
            #    current_accuracy = accuracy.eval(feed_dict={model.x: test_x, model.y: test_y, model.p_keep_hidden: 1.0, model.p_is_training: False})
            #    saver.save(sess, checkpoint_file_path + '/project_tensorflow_car_model_' + str(current_accuracy) +'.ckpt', global_step=step)
            #    print('Saved CheckPoint', str(current_accuracy) )
            if(step % 10 == 0):
                current_accuracys = []
                for number, index in enumerate(validation_batches_list):
                    batch_starting_index = index * batch_size
                    batch_ending_index = min([(index + 1) * batch_size, validation_data_num]) 
                    validation_batch_x = hdf5_file.root.validation_images[batch_starting_index:batch_ending_index]
                    validation_batch_y = hdf5_file.root.validation_labels[batch_starting_index:batch_ending_index]
                    current_accuracys.append(accuracy.eval(feed_dict={model.x: validation_batch_x, model.y: validation_batch_y, model.p_keep_hidden: 1.0, model.p_is_training: False}))
                
                current_accuracy = np.average(current_accuracys)
               
                print('Accuracy %g' % current_accuracy)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    saver.save(sess, checkpoint_file_path + '/project_tensorflow_car_model_' + str(current_accuracy) +'.ckpt', global_step=step)
                    print('Saved CheckPoint', str(current_accuracy) )

    hdf5_file.flush()
    hdf5_file.close()
#tensorboard --logdir=F:/Project_Cars_Data/Logs