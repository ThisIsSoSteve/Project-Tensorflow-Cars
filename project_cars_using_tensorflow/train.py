import os
import tensorflow as tf
import numpy as np
import model
import image


def train_model(number_of_epochs, batch_size, learning_rate, checkpoint_file_path):

    global_step = tf.Variable(0, trainable=False)

    #load data
    training_path = 'training_full.npy'
    training_data = np.load(training_path)

    prediction = model.myModel(model.x, model.z, model.p_keep_hidden)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=model.y))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session() as sess:#config=config
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        while sess.run(global_step) < number_of_epochs:
            epoch_loss = 0
            starting_batch_index = 0
            np.random.shuffle(training_data)
            
            train_x = []
            train_y = []
            train_z = [] #speed
            for data in training_data:#better way?

                train_x.append(np.array(data[0]))
                train_y.append(np.array(data[1]))
                train_z.append(np.array(data[2])) 

            while starting_batch_index < len(training_data[0]):
                start = starting_batch_index
                end = starting_batch_index + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                batch_z = np.array(train_z[start:end])
                batch_z = np.reshape(batch_z ,(batch_size, 1))

                batch_x = batch_x.reshape((-1, model.image_height, model.image_width, 1))

                _, loss_val = sess.run([optimizer, cost], feed_dict = {model.x: batch_x, model.y: batch_y, model.z: batch_z, model.p_keep_hidden: 0.8})
                epoch_loss += loss_val
                starting_batch_index += batch_size

            print('Global Step', sess.run(global_step), 'Loss', epoch_loss)#,'Learning Rate', learning_rate)
        saver.save(sess, checkpoint_file_path)