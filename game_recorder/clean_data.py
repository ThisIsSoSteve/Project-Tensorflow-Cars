import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint

#data/Project_Cars_2017-04-30_10-39-05

path = 'data/Project_Cars_2017-05-20_14-24-50/'

def put_training_data_into_one_file():
    print('put_training_data_into_one_file - Starting')

    training_data = []

    path_training = path + 'training_full.npy'
    if os.path.exists(path_training):
        os.remove(path_training)

    #Load
    for filename in tqdm(os.listdir(path)):
        data = np.load(path + filename)
        #print(data[1])

        #label = data[1] 
        #steering_left = 0.0; 
        #steering_right = 0.0; 

        ##split steering data from (-1 to 1) to (left 0 to 1) and (right 0 to 1)
        #if label[2] > 0:
        #    steering_right = label[2]
        #else:
        #    steering_left = np.absolute(label[2])

        #label = np.array([label[0], label[1], steering_left, steering_right]) #throttle, brakes, left, right
        training_data.append(data) #image, labels

    np.save(path_training, training_data)
    print('put_training_data_into_one_file - Complete')

#put_training_data_into_one_file()

#def balance_data():

#    training_path = path + 'training_full.npy'
#    training_data = np.load(training_path)

#    #throttle=[]
#    #brakes=[]
#    #steer_left=[]
#    #steer_right=[]
#    max_loops = 100000
#    current_loop = 0
#    total_choices = np.array([0.0, 0.0, 0.0, 0.0])
    
#    new_training_data = []
#    #reduce throttle
#    for data in training_data:
#        img = data[0]
#        choices = data[1]
#        if choices[2] < 0.1 and choices[3] < 0.1 and choices[0] > 0.9:
#            if randint(0,32) == 0:
#                new_training_data.append([img, choices])
#            continue
#        elif choices[2] > 0.2 or choices[3] > 0.2:#mirror steering 
#            img = np.fliplr(img)
#            choices = np.array([choices[0], choices[1], choices[3], choices[2]])
#            new_training_data.append([img, choices])
            

#        new_training_data.append([img, choices])

#    training_data = new_training_data
#    #get total
#    for data in training_data:
#        #img = data[0]
#        choices = data[1]
#        total_choices = total_choices + choices

    
#    print('throttle:', total_choices[0], 'brake:', total_choices[1], 'left:', total_choices[2], 'right:', total_choices[3])
#    #throttle: 8264.02303342 brake: 796.199235596 left: 205.753476897 right: 500.607292515

#    min_index = np.argmin(total_choices)
#    min_choice_count = total_choices[min_index]
#    print('min choice count:', min_choice_count)

#    while current_loop < max_loops:
#        random_index = randint(0, len(training_data)-1)

#        randomly_selected_record = training_data[random_index]

#        #if removed does this help
#        the_choice = randomly_selected_record[1]
#        new_total = total_choices - the_choice

#        if new_total[0] >= min_choice_count and new_total[1] >= min_choice_count and new_total[2] >= min_choice_count and new_total[3] >= min_choice_count:
#            training_data = np.delete(training_data, random_index,0)
#            total_choices = total_choices - the_choice
#        #else:
#            #break
#        print('loop:', current_loop)
#        current_loop += 1
        
#    final_total = np.array([0.0, 0.0, 0.0, 0.0])
#    for data in training_data:
#        choices = data[1]
#        final_total = total_choices + choices
#    print('final')
#    print('throttle:', final_total[0], 'brake:', final_total[1], 'left:', final_total[2], 'right:', final_total[3])

#    #print('original total:', len(training_data))
#    #print('throttle:', len(throttle), 'brake:', len(brakes), 'left:', len(steer_left), 'right:', len(steer_right))
#    #print('post total:', len(throttle) + len(brakes) + len(steer_left) + len(steer_right))

#    #choice_counts = np.array([len(throttle), len(brakes), len(steer_left), len(steer_right)])
#    #min_index = np.argmin(choice_counts)
#    #min_choice_count = choice_counts[min_index]

#    #print('min choice count:', min_choice_count)
#    #throttle = throttle[:min_choice_count]
#    #brakes = brakes[:min_choice_count]
#    #steer_left = steer_left[:min_choice_count]
#    #steer_right = steer_right[:min_choice_count]

#    #final_data = throttle + brakes + steer_left + steer_right

#    np.random.shuffle(training_data)
#    np.save(path + 'training_balance_data.npy', training_data)

#balance_data()

def convert_training_data_into_binary():
    print('convert_training_data_into_binary - Starting')

    #training_path = path + 'training_full.npy'
    training_path = path + 'training_balance_data.npy'
    training_data = np.load(training_path)
    np.random.shuffle(training_data)

    writer = tf.python_io.TFRecordWriter("data/project_cars_training_data.tfrecords")
    #WIDTH = 128
    #HEIGHT = 72
    for data in tqdm(training_data):
        features = data[0].flatten()# needs reshaping when use conv2d
        label = data[1]#.astype('float32')

        #print(label.shape)

        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': tf.train.Feature(
                    float_list=tf.train.FloatList(value=label)),#Int64List
                'image': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[features.tobytes()])),#smaller training file size
                    #int64_list=tf.train.Int64List(value=features.astype('int64'))),
        }))

        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    writer.close()
    print('convert_training_data_into_binary - Complete')

convert_training_data_into_binary()

def test_reading_binary_data():
    traning_path = 'data/project_cars_training_data.tfrecords'
    for data in tf.python_io.tf_record_iterator(traning_path):
        example = tf.train.Example()
        example.ParseFromString(data)
        image = example.features.feature['image'].bytes_list.value[0]
        #image = example.features.feature['image'].int64_list.value
        label = example.features.feature['label'].float_list.value
        
        
        test = np.fromstring(image, dtype=np.uint8)
        test = np.reshape(test, (72, 128))
        print(test.shape)


        print('label', np.array(label).shape)
        ##working with Int64List
        #test = np.reshape(image, (72, 128))
        #print(test.shape)

        #plt.imshow(test, cmap='Greys_r')


        plt.matshow(test, cmap=plt.cm.gray)
        plt.show()
        break
        #https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

#test_reading_binary_data()

'''
# get single examples
label, image = read_and_decode_single_example("mnist.tfrecords")
image = tf.cast(image, tf.float32) / 255.
# groups examples into batches randomly
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)

# simple model
w = tf.get_variable("w1", [28*28, 10])
y_pred = tf.matmul(images_batch, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

while True:
  # pass it in through the feed_dict
  _, loss_val = sess.run([train_op, loss_mean])
  print loss_val
'''