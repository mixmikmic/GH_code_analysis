import numpy as np
from pprint import pprint
import scipy.linalg as linalg

np.set_printoptions(precision=3, suppress=True)

a = np.array([[2,5], [4,1]])
#a = np.array([[3,2/3], [2/3,2]])
l,P = np.linalg.eig(a)
pprint(l)
pprint(P)

v0 = P[:,0]
v1 = P[:,1]
pprint(v0)
pprint(v1)

import matplotlib.pyplot as plt

div = 32
for i in range(0,div):
    theta = 2*np.pi/div*i
    x = np.sin(theta)
    y = np.cos(theta)
    # print('%10.5f-%10.5f' % (x,y))
    plt.plot(x,y,'o',color='r')
    p0 = np.array([x,y])
    p1 = np.dot(a,p0)
    plt.plot(p1[0],p1[1],'o',color='b')
    plt.plot([x, p1[0]], [y,p1[1]], color='k', linestyle='-', linewidth=1)

x_m = 7
y_m = 5
plt.hlines(0, -x_m, x_m, color='k', linestyle='-', linewidth=1)
plt.vlines(0, -y_m, y_m, color='k', linestyle='-', linewidth=1)

t=x_m
plt.plot([-t*v0[0],t*v0[0]], [-t*v0[1],t*v0[1]], color='g', linestyle='-', linewidth=2)
t=y_m
plt.plot([-t*v1[0],t*v1[0]], [-t*v1[1],t*v1[1]], color='g', linestyle='-', linewidth=2)


plt.axes().set_aspect('equal', 'datalim')
plt.show()



