import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 



def check_data():
    path = 'data/Project_Cars_2017-05-20_14-24-50/'

    training_path = path + 'training_full.npy'
    training_data = np.load(training_path)

    throttle=[]
    brakes=[]
    steer_left=[]
    steer_right=[]

    for data in training_data:
        img = data[0]
        choices = data[1]
        #print('left', choices[3])

        if choices[3] > 0.8:#left
            steer_left.append([img,choices])
            continue
        if choices[0] < 0.1  and choices[3] > 0.8:#right
            steer_right.append([img,choices])
            continue
        if choices[0] < 0.1 and choices[1] > 0.8 and choices[2] < 0.1 and choices[3] < 0.1:#brake
            brakes.append([img,choices])
            continue
        if choices[0] > 0.1 and choices[1] < 0.1 and choices[2] < 0.1 and choices[3] < 0.1:#throttle
            throttle.append([img,choices])
            continue
            
        
    print('throttle:', len(throttle), 'brake:', len(brakes), 'left:', len(steer_left), 'right:', len(steer_right))



check_data()