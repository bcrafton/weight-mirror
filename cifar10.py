
import numpy as np
import argparse
import keras

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--l2', type=float, default=1e-3)
args = parser.parse_args()

LAYER1 = 1024 * 3
LAYER2 = 1000
LAYER3 = 10

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

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.reshape(TRAIN_EXAMPLES, 1024 * 3)
x_train = x_train.astype('float32')

y_test = keras.utils.to_categorical(y_test, 10)
x_test = x_test.reshape(TEST_EXAMPLES, 1024 * 3)
x_test = x_test.astype('float32')

mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train, axis=0, ddof=1, keepdims=True)
scale = std + 1.
x_train = x_train - mean
x_train = x_train / scale

mean = np.mean(x_test, axis=0, keepdims=True)
std = np.std(x_test, axis=0, ddof=1, keepdims=True)
scale = std + 1.
x_test = x_test - mean
x_test = x_test / scale

#######################################

# high = 1. / np.sqrt((LAYER1 + LAYER2) / 2.)
high = 1. / np.sqrt(LAYER1)
weights1 = np.random.uniform(low=-high, high=high, size=(LAYER1, LAYER2))
bias1 = np.zeros(shape=LAYER2)

# high = 1. / np.sqrt(LAYER2)
high = 1. / np.sqrt(LAYER2)
weights2 = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))
bias2 = np.zeros(shape=LAYER3)

# high = 1. / np.sqrt(LAYER3)
high = 1. / np.sqrt(LAYER2)
b2 = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))

# b2 = np.copy(weights2)

#######################################

for epoch in range(args.epochs):
    
    for ex in range(0, TRAIN_EXAMPLES, 50):
        start = ex 
        stop = ex + 50
    
        A1 = x_train[start:stop]
        Z2 = np.dot(A1, weights1) 
        A2 = Z2 # tanh(Z2)
        Z3 = np.dot(A2, weights2) 
        A3 = softmax(Z3)
        
        labels = y_train[start:stop]
        
        D3 = A3 - labels 
        D2 = np.dot(D3, b2.T) # * dtanh(A2)
        
        # DW2 = np.dot(np.transpose(A2), D3) 
        # DW1 = np.dot(np.transpose(A1), D2)  
        
        # A or Z ? To do X.T * Y.
        # DFB = np.dot(A2.T / np.max(A2), Z3 / np.max(Z3))
        # DFB = np.dot(A2.T, Z3)
        DFB = np.dot(A2.T, Z3)
        
        # weights2 = weights2 - args.lr * DW2 - args.l2 * weights2
        # weights1 = weights1 - args.lr * DW1 - args.l2 * weights1
        
        # DFB = DFB * args.lr * (np.std(DW2) / np.std(DFB))
        # l2 = (args.l2 * b2)
        # b2 = b2 + DFB - l2

        b2 = b2 - (1e-3 * DFB) + (1e-3 * b2)
        print (np.std(weights2), np.std(b2), np.std(DFB))

    # print (np.average(np.absolute(weights2)), np.average(np.absolute(b2)), np.average(np.absolute(DFB)))
    # print (np.std(weights2), np.std(b2), np.std(DFB))
    
    # lol this didnt work well.
    # b2 = b2 / np.std(weights2)
    
    ##################################################
        
    if np.shape(weights2) == np.shape(b2):
        flat_b2 = np.reshape(b2, (-1))
        flat_w2 = np.reshape(weights2, (-1))
    else:
        flat_b2 = np.reshape(b2.T, (-1))
        flat_w2 = np.reshape(weights2, (-1))
        
    angle = angle_between(flat_b2, flat_w2) * (180.0 / 3.14) 

    ##################################################

    correct = 0
    for ex in range(0, TEST_EXAMPLES, 50):
        start = ex 
        stop = ex + 50
    
        A1 = x_test[start:stop]
        Z2 = np.dot(A1, weights1) 
        A2 = Z2 # tanh(Z2)
        Z3 = np.dot(A2, weights2) 
        A3 = softmax(Z3)
        
        labels = y_test[start:stop]
        correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
            
    test_acc = 1. * correct / TEST_EXAMPLES
    
    print ("epoch: %d/%d test acc: %f angle: %f" % (epoch, args.epochs, test_acc, angle))
    
    
    
    
    
