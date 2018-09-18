import numpy as np

training_data_save_path = 'F:/Project_Cars_Data/Training_none_image'
training_data = np.load(training_data_save_path + '/training.npy')

train_x = []
train_y = []

for data in training_data:#better way?
    train_x.append(np.array(data[0]))
    train_y.append(np.array(data[1]))


print(train_x[0])

mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)

print(mean)
print(std)

train_x = (train_x - mean) / std

print(mean.shape)
print(std.shape)

# features = np.array(features)
print( np.array(train_x).shape)

print(train_x[0])


# a = np.array([[1, 2], [3, 4]])
# print(a.shape)