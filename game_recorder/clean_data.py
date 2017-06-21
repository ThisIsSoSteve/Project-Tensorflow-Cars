import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import tensorflow as tf
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from random import randint

#data/Project_Cars_2017-04-30_10-39-05

path = 'data/Project_Cars_2017-06-18_12-58-05/'

def mirror_data(traning_data_to_mirror):

    data = traning_data_to_mirror

    img = data[0]
    label = data[1]

    img = np.fliplr(img)

    choices = np.array([label[0], label[1], label[3], np.absolute(label[2])])

    return np.array([np.float16(img / 255.0), choices, data[2] / 50.0])

def put_training_data_into_one_file():
    print('put_training_data_into_one_file - Starting')

    training_data = []

    path_training = path + 'training_full.npy'
    if os.path.exists(path_training):
        os.remove(path_training)

    #Load
    for filename in tqdm(os.listdir(path)):
        data = np.load(path + filename)

        label = data[1] 
        label = np.array([label[0], label[1], np.absolute(label[2]), label[3]]) #throttle, brakes, left, right

        training_data.append([np.float16(data[0] / 255.0), label, data[2] / 50.0]) #image, labels, speed

        #mirror
        #image = np.fliplr(data[0])
        #label = data[1]
        #label = np.array([label[0], label[1], label[3]], np.absolute(label[2]))
        #training_data.append([np.float16(data[0] / 255.0), label, data[2] / 50.0])

        #training_data.append(mirror_data(data))

    np.save(path_training, training_data)
    print('put_training_data_into_one_file - Complete')

put_training_data_into_one_file()

#def convert_training_data_into_binary():
#    print('convert_training_data_into_binary - Starting')

#    #training_path = path + 'training_full.npy'
#    training_path = path + 'training_balance_data.npy'
#    training_data = np.load(training_path)
#    np.random.shuffle(training_data)

#    writer = tf.python_io.TFRecordWriter("data/project_cars_training_data.tfrecords")
#    #WIDTH = 128
#    #HEIGHT = 72
#    for data in tqdm(training_data):
#        features = data[0].flatten()# needs reshaping when use conv2d
#        label = data[1]#.astype('float32')

#        #print(label.shape)

#        example = tf.train.Example(
#            # Example contains a Features proto object
#            features=tf.train.Features(
#              # Features contains a map of string to Feature proto objects
#              feature={
#                # A Feature contains one of either a int64_list,
#                # float_list, or bytes_list
#                'label': tf.train.Feature(
#                    float_list=tf.train.FloatList(value=label)),#Int64List
#                'image': tf.train.Feature(
#                    bytes_list=tf.train.BytesList(value=[features.tobytes()])),#smaller training file size
#                    #int64_list=tf.train.Int64List(value=features.astype('int64'))),
#        }))

#        # use the proto object to serialize the example to a string
#        serialized = example.SerializeToString()
#        # write the serialized object to disk
#        writer.write(serialized)
#    writer.close()
#    print('convert_training_data_into_binary - Complete')
#convert_training_data_into_binary()




def filter_images():
    training_path = path + 'training_full.npy'
    training_data = np.load(training_path)
    np.random.shuffle(training_data)

    lower_limit = 40
    higher_limit = 160

    amount_of_pictures = 30
    
    for x in range(amount_of_pictures):
        data =  training_data[x]
        print('speed:', data[2])
        #plt.matshow(data[0], vmin=0, vmax=255, cmap=plt.cm.gray)
        #plt.savefig('E:/repos/pics/original/image' + str(x) + '.png')
        #plt.close()
        #plt.clf()

    #for x in range(amount_of_pictures):
    #    data =  training_data[x]
        

    #    data[0][data[0] < lower_limit] = 0
    #    data[0][data[0] > higher_limit] = 0


        #plt.matshow(data[0], vmin=0, vmax=255, cmap=plt.cm.gray)
        #plt.savefig('E:/repos/pics/filtered/image' + str(x) + '.png')
        #plt.close()
        #plt.clf()

    #plt.matshow(data[0], cmap=plt.cm.gray)
    #plt.show()

#filter_images()

#region old stuff
#def test_reading_binary_data():
#    traning_path = 'data/project_cars_training_data.tfrecords'
#    count = 0
#    for data in tf.python_io.tf_record_iterator(traning_path):
#        example = tf.train.Example()
#        example.ParseFromString(data)
#        image = example.features.feature['image'].bytes_list.value[0]
#        #image = example.features.feature['image'].int64_list.value
#        label = example.features.feature['label'].float_list.value
        
        
#        test = np.fromstring(image, dtype=np.uint8)
#        test = np.reshape(test, (72, 128))
#        print(test.shape)


#        print('label shape:', np.array(label).shape, 'label values:', label)
#        ##working with Int64List
#        #test = np.reshape(image, (72, 128))
#        #print(test.shape)

#        #plt.imshow(test, cmap='Greys_r')


#        plt.matshow(test, cmap=plt.cm.gray)
#        plt.show()
#        if count == 30:
#            break
#        else:
#            count += 1
#        #https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

#test_reading_binary_data()





#img = tf.Variable(np.zeros((128, 100)))
#Z = tf.Variable(np.ones((128, 1)))

#with tf.Session() as sess:

#    sess.run(tf.local_variables_initializer())
#    sess.run(tf.global_variables_initializer())

#    all = tf.concat([img, Z], 1)

#    print(sess.run(tf.shape(all)))



#endregion

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