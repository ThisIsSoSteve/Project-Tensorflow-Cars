import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 



def check_data():
    path = 'data/Project_Cars_2017-05-23_22-23-00/'

    training_path = path + 'training_full.npy'
    training_data = np.load(training_path)

    throttle = []
    brakes = []
    steer_left = []
    steer_right = []

    print('Total:', len(training_data))

    for data in training_data:
        img = data[0]
        choices = data[1]
        #print('left', choices[3])

        if choices[2] > 0.9:#left
            steer_left.append([img,[0, choices[1], choices[2], choices[3]]])
            continue
        if choices[3] > 0.9:#right
            steer_right.append([img,[0, choices[1], choices[2], choices[3]]])
            continue
        if choices[0] < 0.1 and choices[1] > 0.8 and choices[2] < 0.1 and choices[3] < 0.1:#brake
            brakes.append([img,choices])
            continue
        if choices[0] > 0.8 and choices[1] < 0.1 and choices[2] < 0.1 and choices[3] < 0.1:#throttle
            throttle.append([img,choices])
            continue
            
    
    print('throttle:', len(throttle), 'brake:', len(brakes), 'left:', len(steer_left), 'right:', len(steer_right))

    #balance left and right
    temp_steer_right = []
    for data in steer_left:
        img = data[0]
        choices = data[1]

        img = np.fliplr(img)
        choices = np.array([choices[0], choices[1], choices[3], choices[2]])
        temp_steer_right.append([img,choices])

    temp_steer_left = []
    for data in steer_right:
        img = data[0]
        choices = data[1]

        img = np.fliplr(img)
        choices = np.array([choices[0], choices[1], choices[3], choices[2]])
        temp_steer_left.append([img,choices])
    
    steer_right = steer_right + temp_steer_right#np.concatenate((steer_right, temp_steer_right))
    steer_left = steer_left + temp_steer_left#np.concatenate((steer_left, temp_steer_left))

    temp_steer_right = None
    temp_steer_left = None

    print('After left right balance')
    print('throttle:', len(throttle), 'brake:', len(brakes), 'left:', len(steer_left), 'right:', len(steer_right))

    #remove throttle 
    temp_throttle = []
    np.random.shuffle(throttle)
    for x in range(0, len(steer_left)):
        temp_throttle.append(throttle[x])

    throttle = temp_throttle
    temp_throttle = None

    print('After throttle right balance')
    print('throttle:', len(throttle), 'brake:', len(brakes), 'left:', len(steer_left), 'right:', len(steer_right))


    print('throttle:', np.array(throttle[0][0]).shape, 'brake:', np.array(brakes[0][0]).shape, 'left:', np.array(steer_left[0][0]).shape, 'right:', np.array(steer_right[0][0]).shape)

    final_data = throttle + brakes + steer_left + steer_right

    np.random.shuffle(final_data)
    np.save(path + 'training_balance_data.npy', final_data)
    print('done total', len(final_data))

check_data()