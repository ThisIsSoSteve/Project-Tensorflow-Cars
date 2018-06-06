import numpy as np
import os
import cv2
import glob
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copy

def convert_raw_to_file(raw_save_path, training_save_path, shuffle):
    print('starting')
     #Setup

    path_training = training_save_path + '/training.npy' 
    if os.path.exists(path_training):
        os.remove(path_training)

    path_training_test = training_save_path + '/training_validation.npy'
    if os.path.exists(path_training_test):
        os.remove(path_training_test)

    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)

    training_data_array = raw_to_array(raw_save_path, 128, 72)

    if shuffle:
        np.random.shuffle(training_data_array)

    #Split validation set from training data
    percent_of_test_data = int((len(training_data_array) / 100) * 20) #20%
    validation_data_array = np.array(training_data_array[0:percent_of_test_data])

    training_data_array = np.array(training_data_array[percent_of_test_data:])

    np.save(path_training, training_data_array)
    np.save(path_training_test, validation_data_array)

    print('Complete')

def mirror_data(image, label):

    image = np.fliplr(image)

    #choices = np.array([label[0], label[1], label[3], label[2]])

    #cv2.imshow("image", image);
    #cv2.waitKey();
    
    #return np.array([image, choices])
    return np.array([image, label])

def raw_to_array(raw_save_path, image_height, image_width):
    print('getting raw data')

    listing = glob.glob(raw_save_path + '/*.png')
    training_data_array = []

    #current = 0

    for filename in tqdm(listing):

        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')

        #Get labels
        project_cars_state = None
        controller_state = None

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        #convert image
        gray_image = cv2.imread(filename + '-image.png', cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR)#cv2.IMREAD_GRAYSCALE
        gray_image = cv2.resize(gray_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC) #keep 16:9 ratio (width, height)
        gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0
        gray_image = gray_image.reshape(image_height, image_width, 1)

        #label = get_throttle_brakes_steering_label(controller_state)
        label = get_is_car_on_track_label(project_cars_state)

        training_data_array.append([gray_image, label])
        #training_data_array.append(mirror_data(gray_image, label))
        #if current > 1000:
        #    break
        #current += 1

   
    print('total data records', len(training_data_array)) 
    return training_data_array

def get_throttle_brakes_steering_label(controller_state):
    throttle = controller_state['right_trigger'] #0 - 255
    brakes = controller_state['left_trigger'] #0 - 255
    steering = controller_state['thumb_lx'] #-32768 - 32767

    steering_left = 0
    steering_right = 0

    if steering < 0:
        steering_left = np.absolute(steering)
        steering_right = 0
    else:
        steering_right = steering
        steering_left = 0

    return np.float16([throttle / 255, brakes / 255, steering_left / 32768, steering_right / 32767]) #throttle, brakes, left, right

def get_is_car_on_track_label(project_cars_state):
    if project_cars_state.mTerrain[0] > 4 or project_cars_state.mTerrain[1] > 4 or project_cars_state.mTerrain[2] > 4 or project_cars_state.mTerrain[3] > 4:
        return np.float16([1, 0])
    else:
        return np.float16([0, 1])

    #print('[{}][{}]'.format(game.mTerrain[0],game.mTerrain[1]))
    #print('[{}][{}]'.format(game.mTerrain[2],game.mTerrain[3]))


def copy_specific_training_data_to_new_folder(source_folder_path, destination_folder_path, track_name, track_variation):
    listing = glob.glob(source_folder_path + '/*.png')
    for filename in tqdm(listing):

        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        #only do Watkins Glen International track data
        current_track = str(project_cars_state.mTrackLocation).replace("'","").replace("b","")
        current_track_variation = str(project_cars_state.mTrackVariation).replace("'","").replace("b","")

        if(current_track != track_name and current_track_variation != track_variation):#if not on the correct track goto next track. *variation = #Short Circuit or #Grand Prix
            continue

        copy(filename + '-data.pkl', destination_folder_path)
        copy(filename + '-image.png', destination_folder_path)




def get_steering_features_labels(raw_save_path, path_training, image_height, image_width):

    listing = glob.glob(raw_save_path + '/*.png')
    np.random.shuffle(listing)

    training_data_final= []
    training_data_left = []
    training_data_right = []
    training_data_no_turns = []

    validation_data_final = []
    validation_data_left = []
    validation_data_right = []
    validation_data_no_turns = []


    buffer = 10000
    limit = 30000
    test_set_limit = limit * 0.3
    currentcount = 0 
    cropped_pixels = int((image_width - image_height) / 2)

    left_turns = 0
    right_turns = 0
    no_turns = 0

    start_adding_data_to_validation_set = False


    for filename in tqdm(listing):

        if currentcount >= limit:
            start_adding_data_to_validation_set = True

        filename = filename.replace('\\','/')
        filename = filename.replace('-image.png','')
        #print(filename)

        with open(filename + '-data.pkl', 'rb') as input:
            project_cars_state = pickle.load(input)
            controller_state = pickle.load(input)

        #only do Watkins Glen International track data
        current_track = str(project_cars_state.mTrackLocation).replace("'","").replace("b","")
        current_track_variation = str(project_cars_state.mTrackVariation).replace("'","").replace("b","")

        if(current_track != 'Watkins Glen International' and current_track_variation != 'Short Circuit'):#if not on the correct track goto next track. *variation = #Short Circuit or #Grand Prix
            continue

        gray_image = cv2.imread(filename + '-image.png', cv2.IMREAD_GRAYSCALE)

        # cv2.imshow("image", gray_image)
        # cv2.waitKey()

        gray_image = cv2.resize(gray_image, (image_width, image_height))
        gray_image = np.float16(gray_image / 255.0) #0-255 to 0.0-1.0

        #cropped width img[y:y+h, x:x+w]
        #gray_image = gray_image[0:image_height, cropped_pixels: cropped_pixels + image_height]

        # #mirror data
        # gray_image_mirror = np.fliplr(gray_image)
        # label_mirror = label

        # if label[1] == 1.0:
        #     label_mirror = np.array([0.0, 0.0, 1.0])
        # elif label[2] == 1.0:
        #     label_mirror = np.array([0.0, 1.0, 0.0])
        

        #print(gray_image.shape)

        #gray_image = gray_image.reshape(image_height, image_width, 1)

        # pic = np.uint8(gray_image * 255.0)
        # plt.matshow(pic, cmap=plt.cm.gray)
        # plt.show()

        # pic = np.uint8(gray_image_mirror * 255.0)
        # plt.matshow(pic, cmap=plt.cm.gray)
        # plt.show()


        label = np.zeros([3])

        current_steering_state = controller_state['thumb_lx']   

        #print(current_steering_state) 
       
        if current_steering_state > 0:
            label = np.array([0.0, 0.0, 1.0])#right
            right_turns += 1
            #print("right")
            if start_adding_data_to_validation_set:
                validation_data_right.append([gray_image, label])
            else:
                training_data_right.append([gray_image, label])

        else:
            label = np.array([0.0, 1.0, 0.0])#left
            left_turns += 1
            #print("left")
            if start_adding_data_to_validation_set:
                validation_data_left.append([gray_image, label])
            else:
                training_data_left.append([gray_image, label])

        if current_steering_state < buffer and current_steering_state > -buffer:
            label = np.array([1.0, 0.0, 0.0])#no input
            no_turns += 1
            if start_adding_data_to_validation_set:
                validation_data_no_turns.append([gray_image, label])
            else:
                training_data_no_turns.append([gray_image, label])




        #previous_steering_state = current_steering_state
        

            #test_data_array.append([gray_image, label])
            #test_data_array.append([gray_image_mirror, label_mirror])
        #else:
            #training_data_array.append([gray_image, label])
            #training_data_array.append([gray_image_mirror, label_mirror])

       

        currentcount += 1

        if currentcount >= limit + test_set_limit:
            break

    training_counts = np.array([len(training_data_right), len(training_data_left), len(training_data_no_turns)])
    min_training_counts = np.argmin(training_counts)
    print(training_counts)

    index_limit = training_counts[min_training_counts]
    print(index_limit)

    for data in training_data_right[0:index_limit]:#better way?
        training_data_final.append([data[0], data[1]])
    for data in training_data_left[0:index_limit]:#better way?
        training_data_final.append([data[0], data[1]])
    for data in training_data_no_turns[0:index_limit]:#better way?
        training_data_final.append([data[0], data[1]])

    validation_counts = np.array([len(validation_data_right), len(validation_data_left), len(validation_data_no_turns)])
    min_validation_counts = np.argmin(validation_counts)
    print(validation_counts)

    index_limit = validation_counts[min_validation_counts]
    print(index_limit)

    for data in validation_data_right[0:index_limit]:#better way?
        validation_data_final.append([data[0], data[1]])
    for data in validation_data_left[0:index_limit]:#better way?
        validation_data_final.append([data[0], data[1]])
    for data in validation_data_no_turns[0:index_limit]:#better way?
        validation_data_final.append([data[0], data[1]])

    print('left_turns: {} right_turns: {} no_turns: {}'.format(left_turns, right_turns, no_turns))
    print('training_data_final_length: {} validation_data_final_length: {} '.format(len(training_data_final), len(validation_data_final)))
    np.save(path_training + '/training.npy' , training_data_final)
    np.save(path_training + '/training_validation.npy' , validation_data_final)




#copy_specific_training_data_to_new_folder('F:/Project_Cars_Data/Raw', 'F:/Project_Cars_Data/Watkins Glen International - Short Circuit', 'Watkins Glen International', 'Short Circuit')

# b'Watkins Glen International'
# b'Short Circuit'

# b'Watkins Glen International'
# b'Grand Prix'
