from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

# load image using PIL
img = Image.open("7.png")
# convert to black and white
img = img.convert("L")
# convert to numpy array
img = np.asarray(img)
# display
get_ipython().run_line_magic('matplotlib', 'inline')
imshow(img, cmap='gray')

print(img.shape)
img

from QNN.layers import *
import pickle

qnn = pickle.load(open("mnist-w1a1.pickle", "rb"))
qnn

# get the predictions array
res = predict(qnn, img)
# return the index of the largest prediction
winner_ind = np.argmax(res)
# the sum of the output values add up to 1 due to softmax,
# so we can interpret them as probabilities
winner_prob = 100 * res[winner_ind]
print(res)
print("The QNN predicts this is a %d with %f percent probability" % (winner_ind, winner_prob))

