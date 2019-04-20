
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

def zca_approx(data, ksize, ssize):
    N, H, W, C = np.shape(data)
    KX, KY, KZ = ksize
    SX, SY, SZ = ssize

    for sx in range(0, KX, SX):
        for sy in range(0, KY, SY):
            for sz in range(0, KZ, SZ):
            
                for x in range(sx, H+sx, KX):
                    for y in range(sy, W+sy, KY):
                        for z in range(sz, C+sz, KZ):
                            
                            x1 = x
                            x2 = x + KX

                            y1 = y
                            y2 = y + KY

                            z1 = z
                            z2 = z + KZ

                            if (x2 > H or y2 > W or z2 > C):
                                continue

                            print (x, y, z)
                
                            white = whiten(X=x_train[:, x1:x2, y1:y2, z1:z2], method='zca')
                            white = np.reshape(white, (N, x2-x1, y2-y1, z2-z1))
                            x_train[:, x1:x2, y1:y2, z1:z2] = white

    return x_train
                            
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


########
# x_train = whiten(x_train)
x_train = zca_approx(x_train, (8, 8, 3), (8, 8, 3))

x_train = np.reshape(x_train, (50000, 1024 * 3))
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







