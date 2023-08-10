import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
get_ipython().run_line_magic('matplotlib', 'inline')

def normalize(ln):
    if ln[0,0] < ln[0,1]:
        return ln[:,::-1]
    else:
        return ln

def get_m(x):
    dy = x[1][1] - x[1][0]
    dx = x[0][1] - x[0][0]
    return dy / dx

def get_y(ln):
    m = get_m(ln)
    b = ln[1][1]
    return lambda x: m * x + b

def fit_line(ln):
    y = get_y(normalize(ln))
    return np.r_[300,0,y(300),y(0)].reshape(2,2)

matplotlib.rcParams['image.cmap'] = 'Greys_r'
matplotlib.rcParams['figure.figsize'] = [10,15] 

i = 4

i = i-4

i = i+1

lns = loadmat('./annotations/{}.mat'.format(i))['lines']

img = cv2.imread('./bills_cropped/{}.jpg'.format(i),0)[:,:]

lns = map(fit_line, lns)

plt.imshow(img)
for l in lns:
    plt.plot(l[0],l[1],c='r')

print(i)



