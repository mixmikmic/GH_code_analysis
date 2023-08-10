import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from darkflow.defaults import argHandler

from darkflow.net.build import TFNet

FLAGS = argHandler()
FLAGS.setDefaults()

cfgFile = "yolo-voc.cfg"
wFile = "yolo-voc.weights"

if not os.path.exists("/data/"+ cfgFile):
    os.system("wget --directory-prefix=/data https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/" + cfgFile)
if not os.path.exists("/data/" + wFile):
    os.system("wget --directory-prefix=/data https://pjreddie.com/media/files/" + wFile)

FLAGS["model"] = "/data/" + cfgFile
FLAGS["load"] = "/data/" + wFile
FLAGS["summary"] = "/log"

tfnet = TFNet(FLAGS)

darknet = tfnet.darknet

for layer in darknet.layers:
    print layer.type

layer0 = darknet.layers[0]
print(layer0.w['kernel'])

k = layer0.w['kernel']

import tensorflow as tf
session = tfnet.sess

def weightsToImage(tensor):
    kern = session.run(tensor)
    ky, kx, channels, numkernel = kern.shape
    kernReshaped = kern.swapaxes(0,1).swapaxes(0,2).swapaxes(2,3)
    kernFlat = kern.reshape(channels * ky,kx*numkernel)
    #kernFlat = kernFlat.swapaxes(0,1).swapaxes(1,2)
    #print kernFlat.shape
    plt.imshow(kernFlat, interpolation='nearest')
    plt.colorbar()
    plt.show()

for layeridx, layer in enumerate(darknet.layers):
    
    print "Layer idx:%d, type: %s" % (layeridx, layer.type)
    if layer.type == "convolutional":
        tensor = layer.w['kernel']
        ky, kx, chan, kernels = tensor.shape
        print "kernelsize: x:%d, y:%d, channels: %d, number of kernels:%d" % (kx, ky, chan, kernels)
        weightsToImage(tensor)

if not os.path.exists("person.jpg"):
    os.system("wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg")

im = cv2.imread("person.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);

plt.imshow(im)
plt.show()

a = np.asarray(im)

pred = tfnet.return_predict(a)

pred

print im.shape
for p in pred:
    pt1 = (p["bottomright"]["x"],p["bottomright"]["y"])
    pt2 = (p["topleft"]["x"],p["topleft"]["y"])
    if (p["confidence"] > 0.1):
        cv2.rectangle(im, pt1, pt2, (255,0,0), 3)
plt.imshow(im)
plt.show()





