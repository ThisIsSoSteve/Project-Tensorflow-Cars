#import os
import pickle
import glob
#import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#from data_control_no_images.read import Read

listing = glob.glob('F:/Project_Cars_Data/1lap-fullspeed/Watkins Glen International - Short Circuit' + '/*.pkl')

x = []
y = []

throttle = []
raw_throttle = []

brake = []
raw_brake = []

steering = []
raw_steering = []

xy = []

for filename in tqdm(listing):

    with open(filename, 'rb') as file_data:
        project_cars_state = pickle.load(file_data)
        controller_state = pickle.load(file_data)

        #remove none flying lap data
        if project_cars_state.mParticipantInfo[0].mCurrentLapDistance == 0.0:
                continue

        position = project_cars_state.mParticipantInfo[0].mWorldPosition

        x.append(round(position[0]))
        y.append(round(position[2]))
        throttle.append(controller_state['right_trigger']/255)# 0 - 255
        brake.append(controller_state['left_trigger']/255) #0 - 255
        steering.append(controller_state['thumb_lx']/32767) #-32768 - 32767
        #steering.append(project_cars_state.mSteering)
        raw_steering.append(project_cars_state.mUnfilteredSteering)
        raw_brake.append(project_cars_state.mUnfilteredBrake)
        raw_throttle.append(project_cars_state.mUnfilteredThrottle)

        xy.append([position[0], position[2]])

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=steering)
plt.colorbar()
plt.axis('equal')
plt.title('position and controller steering')
plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=raw_steering)
plt.colorbar()
plt.axis('equal')
plt.title('position and raw steering')
plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=throttle)
plt.colorbar()
plt.axis('equal')
plt.title('position and controller throttle')
plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=raw_throttle)
plt.colorbar()
plt.axis('equal')
plt.title('position and raw throttle')
plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=brake)
plt.colorbar()
plt.axis('equal')
plt.title('position and controller brake')
plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=raw_brake)
plt.colorbar()
plt.axis('equal')
plt.title('position and raw brake')
plt.show()
plt.close()

        
# get_data = Read(True)

# mean, std = get_data.load_mean_and_std('F:/Project_Cars_Data/Full_Speed_Training_none_image')

# print(mean)
# print(std)
# xy = (xy - mean) / std

# print(np.array(xy[:,0]).shape)

# plt.scatter(xy[:,0], xy[:,1])
# plt.axis('equal')
# plt.show()
        