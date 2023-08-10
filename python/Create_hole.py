import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.misc import face
from scipy.linalg import norm
from skimage import measure
import networkx as nx
from sklearn import cluster
import threading
import Queue
import time
from multiprocessing import cpu_count
import cv2

imgpath = "../ressources/ski.png"

# For a jpg, jpeg or png picture
img = cv2.imread(imgpath)

# For a picture already in npy
#img = np.load(imgpath)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Resize if needed
# img = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)


plt.imshow(img)
plt.axis("off")
plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()

boxes = []
counter = 0
while True:
    def on_mouse(event, x, y, flags, params):
        global boxes, img, counter

        if event == cv2.EVENT_LBUTTONDOWN:
            #print 'Start Mouse Position: '+str(x)+', '+str(y)
            boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            #print 'End Mouse Position: '+str(x)+', '+str(y)
            ebox = (x, y)
            boxes.append((x, y))
            img = cv2.rectangle(img, boxes[0], boxes[1], [0, 0, 0], thickness=-1)
            boxes = []
            counter += 1

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', on_mouse, 0)
    cv2.imshow('Image',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cv2.imwrite("../ressources/created_.jpg", img)
        break

cv2.imwrite("result_jerusalem.jpg", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
np.save("./Patch-Match/ressources/imagetest/rond_creux.npy", img)

type(range(5))

sum(set(np.arange(0,N,3)).union(set(np.arange(0,N,5))))

a=set([1,2,3,4,5])

N = 50
a = 0
b = 0
result = 0
while (a<N) & (b<N):
    if a<b:
        a += 3
        if a !=b:
            result += a
    else:
        b += 5
        if b != a:
            result += b
print(result)



