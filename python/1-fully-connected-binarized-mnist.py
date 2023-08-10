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
qnn = pickle.load(open("mnist-w1a1.pickle", "rb"))
qnn

import inspect

def showSrc(what):
    print("".join(inspect.getsourcelines(what)[0]))

showSrc(predict)

def simpleThreshold(x):
    if x <= 10:
        return -1
    else:
        return +1

x = range(-50,50)
y = map(simpleThreshold, x)
plt.plot(x, y)
plt.ylim([-5, 5])
plt.show()

binarized_input = qnn[0].execute(img)
plt.imshow(binarized_input.reshape(28,28), cmap='gray')

showSrc(QNNBipolarThresholdingLayer)

showSrc(QNNThresholdingLayer)

print("First thresholding layer, parameter shape: " + str(qnn[0].thresholds.shape))
print("First thresholding layer, threshold values: \n" + str(qnn[0].thresholds))

print("Second thresholding layer, parameter shape: " + str(qnn[2].thresholds.shape))
print("Second thresholding layer, threshold values: \n" + str(qnn[2].thresholds))

showSrc(QNNFullyConnectedLayer)

print("First FC layer weight matrix shape: " + str(qnn[1].W.shape))
qnn[1].W

fc_0_output = predict(qnn[0:2], img)
print("First FC layer output shape: " + str(fc_0_output.shape))
fc_0_output

qnn[0:8]

showSrc(QNNScaleShiftLayer)

print("Scale: " + str(qnn[8].A))
print("Shift: " + str(qnn[8].B))

showSrc(QNNSoftmaxLayer)

ret_no_softmax = predict(qnn[:8], img)
print("Last layer outupts without softmax: " + str(ret_no_softmax))
print("Predicted class: %d" % np.argmax(ret_no_softmax))



