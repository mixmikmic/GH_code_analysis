from neon.data import CIFAR10   # Neon's helper function to download the CIFAR-10 data
from PIL import Image           # The Python Image Library (PIL)
import numpy as np              # Our old friend numpy

dataset = dict()   # We'll create a dictionary for the training and testing/validation set images

pathToSaveData = './path'  # The path where neon will download and extract the data files

cifar10 = CIFAR10(path=pathToSaveData, normalize=False, whiten=False, contrast_normalize=False)
dataset['train'], dataset['validation'], numClasses = cifar10.load_data()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def getImage(dataset=dataset, setName='train', index=0):
    
    # The images are index 0 of the dictionary
    # They are stored as a 3072 element vector so we need to reshape this into a tensor.
    # The first dimension is the red/green/blue channel, the second is the pixel row, the third is the pixel column
    im = dataset[setName][0][index].reshape(3,32,32)
    
    # PIL and matplotlib want the red/green/blue channels last in the matrix. So we just need to rearrange 
    # the tensor to put that dimension last.
    im = np.transpose(im, axes=[1, 2, 0])  # Put the 0-th dimension at the end
    
    # Image are supposed to be unsigned 8-bit integers. If we keep the raw images, then
    # this line is not needed. However, if we normalize or whiten the image, then the values become
    # floats. So we need to convert them back to uint8s.
    im = np.uint8(im)  
    
    classIndex = dataset[setName][1][index][0] # This is the class label (0-9)
    
    classNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Now use PIL's Image helper to turn our array into something that matplotlib will understand as an image.
    return [classIndex, classNames[classIndex], Image.fromarray(im)]

idx, name, im = getImage(dataset, 'train', 1022) # Get image 1022 of the training data.
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'train', 8888)  # Get image 8888 of the training data
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'train', 5002)   # Get image 5002 of the training data
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'train', 10022)  # Get image 10022 of the training data
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'train', 7022)   # Get image 7022 of the training data
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'validation', 1022)  # Get image 1022 of the validation data
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'validation', 1031)   # Get image 1031 of the validation data
plt.imshow(im);
plt.title(name);
plt.axis('off');

idx, name, im = getImage(dataset, 'validation', 9135)  # Get image 9135 of the validation data
plt.imshow(im);
plt.title(name);
plt.axis('off');





