from imagenet_data.classes import *
len(imagenet_classes)
for i in range(10):
    print(imagenet_classes[i])

from QNN.layers import *
import pickle

get_ipython().system('wget -nc http://www.idi.ntnu.no/~yamanu/alexnet-hwgq.pickle')
qnn = pickle.load(open("alexnet-hwgq.pickle", "rb"))

qnn

for i in range(len(qnn)):
    L = qnn[i]
    cname = L.__class__.__name__
    if cname == "QNNConvolutionLayer" or cname == "QNNFullyConnectedLayer":
        print("Weight range for layer %d: min %d max %d" % (i, np.min(L.W), np.max(L.W)))
                

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# define a small utility function to first display, then prepare the
# images for classification
def prepare_imagenet(img):
    # make sure the image is the size expected by the network
    img = img.resize((227, 227))
    display(img)
    # convert to numpy array
    img = np.asarray(img).copy().astype(np.int32)
    # we need the data layout to be (channels, rows, columns)
    # but it comes in (rows, columns, channels) format, so we
    # need to transpose the axes:
    img = img.transpose((2, 0, 1))
        
    # our network is trained with BGR instead of RGB images,
    # so we need to invert the order of channels in the channel axis:
    img = img[::-1, :, :]
    # finally, we need to subtract the mean per-channel pixel intensity
    # since this is how this network has been trained
    img[0] = img[0] - 104
    img[1] = img[1] - 117
    img[2] = img[2] - 123
    return img

# load test images and prepare them
img_grouse = prepare_imagenet(Image.open("imagenet_data/grouse.jpg"))
img_cat = prepare_imagenet(Image.open("imagenet_data/cat.jpg"))
img_husky = prepare_imagenet(Image.open("imagenet_data/husky.jpg"))

def imagenet_predict(img):
    # get the predictions array
    res = predict(qnn, img)
    # return the index of the largest prediction, then use the
    # classes array to map to a human-readable string
    winner_inds_top5 = np.argsort(res)[-5:]
    winner_ind = winner_inds_top5[-1]
    winner_class = imagenet_classes[winner_ind]
    # the sum of the output values add up to 1 due to softmax,
    # so we can interpret them as probabilities
    winner_prob = 100 * res[winner_ind]
    print("The QNN predicts this is a(n) %s (class %d) with %f percent probability" % (winner_class, winner_ind, winner_prob))
    print("Top-5 classes:")
    for i in winner_inds_top5:
        print(imagenet_classes[i])
    print("")

imagenet_predict(img_grouse)
imagenet_predict(img_cat)
imagenet_predict(img_husky)

