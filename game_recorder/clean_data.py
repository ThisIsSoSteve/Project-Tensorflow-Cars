import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint

#data/Project_Cars_2017-04-30_10-39-05

path = 'data/Project_Cars_2017-06-10_21-45-44/'

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
        training_data.append([data[0], label, data[2]]) #image, labels

    np.save(path_training, training_data)
    print('put_training_data_into_one_file - Complete')

#put_training_data_into_one_file()

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
#convert_training_data_into_binary()

def test_reading_binary_data():
    traning_path = 'data/project_cars_training_data.tfrecords'
    count = 0
    for data in tf.python_io.tf_record_iterator(traning_path):
        example = tf.train.Example()
        example.ParseFromString(data)
        image = example.features.feature['image'].bytes_list.value[0]
        #image = example.features.feature['image'].int64_list.value
        label = example.features.feature['label'].float_list.value
        
        
        test = np.fromstring(image, dtype=np.uint8)
        test = np.reshape(test, (72, 128))
        print(test.shape)


        print('label shape:', np.array(label).shape, 'label values:', label)
        ##working with Int64List
        #test = np.reshape(image, (72, 128))
        #print(test.shape)

        #plt.imshow(test, cmap='Greys_r')


        plt.matshow(test, cmap=plt.cm.gray)
        plt.show()
        if count == 30:
            break
        else:
            count += 1
        #https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

#test_reading_binary_data()


img = tf.Variable(np.zeros((128, 100)))
Z = tf.Variable(np.ones((128, 1)))

#test = tf.Variable(np.zeros(101))

with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    #img.assign(np.zeros(100))
    #Z.assign(np.zeros(1))

    #all = []

    #for i in range(128):
    #    realImg = img[i]
    #    realZ = Z[i]
    #    all.append(tf.concat([realImg, [realZ]], 0))

    #all = tf.stack(all)

    all = tf.concat([img, Z], 1)

    print(sess.run(tf.shape(all)))

    #for i in range(128):
        #realImg = sess.run(img[i])
        #realZ = sess.run(Z[i])

        #print(realZ.shape)
       # both = np.concatenate((realImg, [realZ]), axis=0)

       # all.append(both)

        #print(i, both.shape)
    
    

    #testing = tf.stack(all)
    #print(testing)
    
    #newImg = tf.concat([img, Z], 0)

    #shape = sess.run(tf.shape(newImg))
    #Z = sess.run(tf.shape(Z))

    #print("newImg shape:", shape)
    #print("newImg 100:", sess.run(newImg[100]))



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