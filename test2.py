
import numpy as np

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

########

high = 1.
low = -high
N = 1000
M = 1024 * 3
w = np.random.uniform(low=low, high=high, size=(N, M))
flat_w = np.reshape(w, (-1))

########

xy = 0.
for _ in range(1000):
    # x = np.random.uniform(low=low, high=high, size=(M, 1)) 
    x = np.random.normal(loc=0., scale=high, size=(M, 1)) 
    y = w @ x
    xy += y @ x.T

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
