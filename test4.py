
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from whiten import whiten

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
assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = np.transpose(x_train, (0, 3, 1, 2))

mean = np.mean(x_train, axis=(0), keepdims=True)
std = np.std(x_train, axis=(0), ddof=1, keepdims=True)

scale = std 
x_train = x_train - mean
x_train = x_train / scale

x_train = np.reshape(x_train, (50000, 1024 * 3))
x_train = whiten(x_train)
x_train = x_train.astype('float32')

########

xx = 0.
for idx in range(1000):
    print (idx)
    x = np.reshape(x_train[idx], (1024 * 3, 1))
    xx += x @ x.T

xx = xx / np.max(xx)
plt.imshow(xx, cmap='gray')
plt.show()







