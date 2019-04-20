
import numpy as np
import tensorflow as tf

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

########

high = 1.
low = -high
N = 1024 * 3
M = 1000
w = np.random.uniform(low=low, high=high, size=(N, M))
flat_w = np.reshape(w, (-1))

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

mean = np.mean(x_train, axis=(1, 2, 3), keepdims=True)
std = np.std(x_train, axis=(1, 2, 3), ddof=1, keepdims=True)

# print (np.shape(mean), np.shape(std))

# mean = np.mean(x_train)
# std = np.std(x_train, ddof=1)

scale = std 
x_train = x_train - mean
x_train = x_train / scale

x_train = x_train.reshape(50000, 1024 * 3)
x_train = x_train.astype('float32')

# print (np.mean(x_train), np.std(x_train))

########

xy = 0.
for idx in range(1000):
    print (idx)
    # perm = np.random.permutation(1024 * 3)
    # x = np.reshape(x_train[idx][perm], (1024 * 3, 1))
    
    # x = np.random.uniform(low=low, high=high, size=(N, 1)) 
    
    x = np.reshape(x_train[idx], (1024 * 3, 1))
    
    y = w.T @ x
    xy += x @ y.T

xy = xy * (np.std(w) / np.std(xy))

flat_xy = np.reshape(xy, (-1))
angle1 = angle_between(flat_xy, flat_w) * (180.0 / 3.14) 
loss1 = np.sum((w - xy) ** 2)
####
xy = np.random.uniform(low=low, high=high, size=(N, M))

flat_xy = np.reshape(xy, (-1))
angle2 = angle_between(flat_xy, flat_w) * (180.0 / 3.14) 
loss2 = np.sum((w - xy) ** 2)
####

print (angle1, angle2)








