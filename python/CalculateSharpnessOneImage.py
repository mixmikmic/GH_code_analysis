from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import array, int8
from skimage import feature
from scipy.ndimage import convolve
from scipy import misc
from scipy import ndimage

from omero.gateway import BlitzGateway
HOST = 'outreach.openmicroscopy.org'
# To be modified
USERNAME = 'username'
PASSWORD = 'password'
PORT = 4064
conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
conn.connect()

class AlgorithmList:
    def gradientBasedSharpnessMetric(self):
        gy, gx = np.gradient(plane)
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        return sharpness
    def fourierBasedSharpnessMetric(self):
        fftimage = np.fft.fft2(plane)
        fftshift = np.fft.fftshift(fftimage)
        fftshift = np.absolute(fftshift)
        M = np.amax(fftshift)
        Th = (fftshift > (M/float(1000))).sum()
        if 'image' in locals():
            sharpness = Th/(float(image.getSizeX())*float(image.getSizeY()))
            return sharpness*10000
        else:
            return Th
    def edgeBasedSharpnessMetric(self):
        edges1 = feature.canny(plane, sigma=3)
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        sharpness = convolve(edges1, kernel, mode="constant")
        sharpness = sharpness[edges1 != 0].sum()
        return sharpness
print "loaded:", dir(AlgorithmList)

from ipywidgets import widgets
def dropdown_widget(Algorithm_list,
                    dropdown_widget_name,
                    displaywidget=False):

    alg_sel = widgets.Dropdown(
        options=Algorithm_list,
        value=Algorithm_list[0],
        description=dropdown_widget_name,
        disabled=False,
    )
    if displaywidget is True:
        display(alg_sel)
    return alg_sel

Algorithm = dropdown_widget(
    ['Gradient', 'Fourier', 'Edge'],
    'Algorithm: ', True
)
# SELECT THE METHOD THEN MOVE TO THE NEXT CELL WITHOUT RUNNING THE CELL AGAIN

method = Algorithm.value
if method == 'Gradient':
    sharpness_method = AlgorithmList().gradientBasedSharpnessMetric
elif method == 'Fourier':
    sharpness_method = AlgorithmList().fourierBasedSharpnessMetric
elif method == 'Edge':
    sharpness_method = AlgorithmList().edgeBasedSharpnessMetric

resultArray = np.zeros((5, 2), dtype=float)
plt.figure(figsize=(20, 15))
cntr = 1
for sigValue in xrange(0,20,4):
    face = misc.face(gray=True)
    plane = ndimage.gaussian_filter(face, sigma=sigValue)
    plt.subplot(1,5,cntr)
    plt.imshow(plane, cmap=plt.cm.gray)
    plt.axis('off')
    sharpness = sharpness_method()
    resultArray[cntr-1,1] = sharpness
    resultArray[cntr-1,0] = sigValue
    cntr= cntr + 1

plt.show()
plt.figure(figsize=(15, 8))
plt.plot(resultArray[:,0], resultArray[:,1], 'ro')
plt.title(method)
plt.xlabel('Levels of gaussian blur')
plt.ylabel('Sharpness score')
plt.show()
plt.gcf().clear() 

# To be modified
# ex: Select an Image from the dataset named 'CellProfiler images' and enter its Id
imageId = 9397
image = conn.getObject("Image", imageId)
print image.getName(), image.getDescription()

pixels = image.getPrimaryPixels()
image_plane = pixels.getPlane(0, 0, 0)

resultArray = np.zeros((5, 2), dtype=float)
plt.figure(figsize=(20, 15))
cntr = 1
for sigValue in xrange(0, 20, 4):
    face = misc.face(gray=True)
    plane = ndimage.gaussian_filter(image_plane, sigma=sigValue)
    plt.subplot(1, 5, cntr)
    plt.imshow(plane, cmap=plt.cm.gray)
    plt.axis('off')
    sharpness = sharpness_method()
    resultArray[cntr-1, 1] = sharpness
    resultArray[cntr-1, 0] = sigValue
    cntr = cntr + 1

plt.show()
plt.figure(figsize=(15, 8))
plt.plot(resultArray[:,0], resultArray[:,1], 'ro')
plt.title(method)
plt.xlabel('Levels of gaussian blur')
plt.ylabel('Sharpness score')
plt.show()
plt.gcf().clear()

conn.close()

