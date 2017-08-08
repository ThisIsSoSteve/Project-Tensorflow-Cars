import os
import tensorflow as tf
import numpy as np
import model
from tables import *
from math import ceil
import time

def train_model(number_of_epochs, batch_size, learning_rate, training_file_path, checkpoint_file_path, checkpoint_save_path):

    path_training = training_file_path + '/on_track_training.h5'

    global_step = tf.Variable(0, trainable=False)

    prediction = model.myModel(model.x, model.p_keep_hidden)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=model.y))
    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step = global_step)#epsilon =1e-04

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(model.y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    saver = tf.train.Saver(max_to_keep=10)

    best_accuracy = 0.3

    hdf5_file = open_file(path_training, mode='r', driver="H5FD_CORE")

    data_num = hdf5_file.root.training_images.shape[0]
    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    
    validation_data_num = hdf5_file.root.validation_images.shape[0]
    # create validation list of batches
    validation_batches_list = list(range(int(ceil(float(validation_data_num) / batch_size))))

    # create a summary for our cost and accuracy
    tf.summary.scalar("Cross_entropy", cross_entropy)
    tf.summary.scalar("Accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session 
    merged = tf.summary.merge_all()
    #merged = tf.summary.merge([accuracy_sum, cross_entropy_sum])
    #tensorboard --logdir=F:/Project_Cars_Data/Logs/OnTrack
    #F:
    #python -m tensorflow.tensorboard --logdir=Project_Cars_Data/Logs/OnTrack

    print('training number', data_num, 'validation number', validation_data_num)

    with tf.Session() as sess:

        #sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if checkpoint_save_path != '':
            saver.restore(sess, checkpoint_save_path)
        step = sess.run(global_step)

        logs_path = "F:/Project_Cars_Data/Logs/OnTrack"
        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        total = 0

        while step < number_of_epochs:
            epoch_loss = 0
            np.random.shuffle(batches_list)
            
            for number, index in enumerate(batches_list):
                batch_starting_index = index * batch_size
                batch_ending_index = min([(index + 1) * batch_size, data_num]) 

                batch_x = hdf5_file.root.training_images[batch_starting_index:batch_ending_index]
                batch_y = hdf5_file.root.training_labels[batch_starting_index:batch_ending_index]

                _, loss_val, summary = sess.run([optimizer, cross_entropy, merged], feed_dict = {model.x: batch_x, model.y: batch_y, model.p_keep_hidden: 0.8})

                epoch_loss += loss_val
                # write log
                #print(total)
                writer.add_summary(summary, total)
                #writer.flush()
                total += 1

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
                    current_accuracys.append(accuracy.eval(feed_dict={model.x: validation_batch_x, model.y: validation_batch_y, model.p_keep_hidden: 1.0}))
                #print(current_accuracys)

                current_accuracy = np.average(current_accuracys)
                
               
                print('Accuracy %g' % current_accuracy)
                if current_accuracy > best_accuracy or step % 50 == 0:
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    saver.save(sess, checkpoint_file_path + '/project_tensorflow_car_model_' + str(current_accuracy) +'.ckpt', global_step=step)
                    print('Saved CheckPoint', str(current_accuracy) )

    hdf5_file.flush()
    hdf5_file.close()