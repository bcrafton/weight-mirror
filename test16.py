
import numpy as np
import argparse
import keras
import matplotlib.pyplot as plt
from whiten import whiten

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--l2', type=float, default=1e-3)
args = parser.parse_args()

LAYER1 = 100
LAYER2 = 100
LAYER3 = 100
LAYER4 = 10

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000

#######################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
    return x * (1. - x)

def relu(x):
    return x * (x > 0)
  
def drelu(x):
    # USE A NOT Z
    return 1.0 * (x > 0)

def tanh(x):
  return np.tanh(x)
  
def dtanh(x):
  # USE A NOT Z
  return (1. - (x ** 2))

#######################################

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
                
                            white = whiten(X=data[:, x1:x2, y1:y2, z1:z2], method='zca')
                            white = np.reshape(white, (N, x2-x1, y2-y1, z2-z1))
                            data[:, x1:x2, y1:y2, z1:z2] = white

    return data

#######################################
'''
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert(np.shape(x_train) == (50000, 32, 32, 3))
assert(np.shape(x_test) == (10000, 32, 32, 3))

y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.astype('float32')

y_test = keras.utils.to_categorical(y_test, 10)
x_test = x_test.astype('float32')

mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train, axis=0, ddof=1, keepdims=True)
scale = std
x_train = x_train - mean
x_train = x_train / scale
x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
# x_train = whiten(x_train)
# x_train = zca_approx(x_train, (8, 8, 3), (8, 8, 3))
x_train = x_train.reshape(TRAIN_EXAMPLES, 1024 * 3)
'''
#######################################

high = 1. / np.sqrt(LAYER1)
weights0 = np.random.uniform(low=-high, high=high, size=(LAYER1, LAYER1))
# weights0 = np.ones(shape=(LAYER1, LAYER1))
# weights0 = np.eye(LAYER1)
# print (np.linalg.matrix_rank(weights0))
# print (np.linalg.matrix_rank(weights0.T @ weights0))

high = 1. / np.sqrt(LAYER1)
weights1 = np.random.uniform(low=-high, high=high, size=(LAYER1, LAYER2))
bias1 = np.zeros(shape=LAYER2)

high = 1. / np.sqrt(LAYER2)
weights2 = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))
bias2 = np.zeros(shape=LAYER3)

'''
high = 1. / np.sqrt(LAYER3)
weights3 = np.random.uniform(low=-high, high=high, size=(LAYER3, LAYER4))
bias3 = np.zeros(shape=LAYER4)
'''

########

xy = 0.

batch_size = 100
for idx in range(0, 100000, batch_size):
    start = idx
    end = idx + batch_size
    
    x = np.random.uniform(low=-1., high=1., size=(batch_size, LAYER1))

    x = x @ weights0
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, ddof=1, keepdims=True)
    x = (x - mean) / std
    x = whiten(x)
    
    y = x @ weights1
    xy += x.T @ y

########

angle1 = angle_between(np.reshape(xy, -1), np.reshape(weights0.T @ weights0 @ weights1, -1)) * (180.0 / 3.14) 
angle2 = angle_between(np.reshape(xy, -1), np.reshape(weights1, -1)) * (180.0 / 3.14)
print (angle1, angle2)

########


