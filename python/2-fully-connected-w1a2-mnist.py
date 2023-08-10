from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load image using PIL
img = Image.open("7.png")
# convert to black and white
img = img.convert("L")
# convert to numpy array
img = np.asarray(img)
# display
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img, cmap='gray')

from QNN.layers import *
import pickle
# load the qnn
qnn = pickle.load(open("mnist-w1a2.pickle", "rb"))
qnn

print("Threshold parameter shape: " + str(qnn[2].thresholds.shape))
print("Thresholds for the first vector element: " + str(qnn[2].thresholds[:, 0]))
print("Thresholds for the second vector element: " + str(qnn[2].thresholds[:, 1]))

def simpleThreshold(x):
    if x <= 5:
        return 0
    elif x <= 49:
        return 1
    elif x <= 78:
        return 2
    else:
        return 3

x = range(-100,100)
y = map(simpleThreshold, x)
plt.plot(x, y)
plt.ylim([-5, 5])
plt.show()

qnn[0].execute(img)

# get the predictions array
res = predict(qnn, img)
# return the index of the largest prediction
winner_ind = np.argmax(res)
# the sum of the output values add up to 1 due to softmax,
# so we can interpret them as probabilities
winner_prob = 100 * res[winner_ind]
print(res)
print("The QNN predicts this is a %d with %f percent probability" % (winner_ind, winner_prob))

