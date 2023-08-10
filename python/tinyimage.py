import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

get_ipython().magic('matplotlib inline')

face = misc.face(gray=True)
plt.imshow(face, cmap=plt.cm.gray)

face_small = misc.imresize(face, (32,32))
plt.imshow(face_small)

misc.imsave('face.png', face_small)

get_ipython().system(' wget http://www.publicdomainpictures.net/pictures/190000/velka/metal-model-toy-car.jpg')

car = misc.imread('metal-model-toy-car.jpg')
plt.imshow(car)
car_small = misc.imresize(car, (32,32))
plt.imshow(car_small)

misc.imsave('car_small.png', car_small)

