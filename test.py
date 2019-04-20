
import numpy as np

N = 5
M = 10

####
x = np.random.uniform(low=0., high=1., size=(M, 1))
w = np.random.uniform(low=0., high=1., size=(N, M))

y = w @ x
####
xy = y @ x.T
xy = xy * (np.std(w) / np.std(xy))

print (xy)
print (w)

loss1 = np.sum((w - xy) ** 2)
####
xy = np.random.uniform(low=0., high=1., size=(N, M))
loss2 = np.sum((w - xy) ** 2)
####
print (loss1, loss2)
