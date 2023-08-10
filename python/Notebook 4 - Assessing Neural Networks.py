get_ipython().magic('matplotlib inline')
import numpy as np

from lfw_fuel import lfw
from models import seattle_model
from clean import clean

modelA = seattle_model() # Use all the defaults. This already calls compile().

print(modelA.summary())

# Load the data, shuffled and split between train and test sets
(X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = lfw.load_data("deepfunneled")

(X_train, y_train), (X_test, y_test) = clean(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

from keras.callbacks import TensorBoard 
tb = TensorBoard(log_dir='./logs', 
                 write_graph=False, 
                 histogram_freq=1, 
                 write_images=True, 
                 embeddings_freq=0)

from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

batch_size = 128
num_epochs = 8

modelA.fit(X_train, y_train, 
          batch_size = batch_size, 
          epochs = num_epochs,
          verbose = 1, 
          validation_data = (X_test, y_test),
          callbacks = [tb,history])

score = modelA.evaluate(X_test, y_test, verbose=0)

print("-"*40)
print("Seattle Model (%d epochs):"%(num_epochs))
print('Test accuracy: {0:%}'.format(score[1]))

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

yy = history.losses
xx = range(len(yy))
ax.plot(xx,yy)
ax.set_xlabel('Batch')
ax.set_ylabel('Loss')
plt.show()

y_predicted = modelA.predict(X_test)

print("Predicted:")
print("Zeros: %d"%(np.sum(y_predicted<0.4)))
print("Not Sure: %d"%(np.sum(np.logical_and(y_predicted>0.4 , y_predicted<0.6))))
print("Ones: %d"%(np.sum(y_predicted>0.6)))
print("\n")
print("Actual:")
print("Zeros: %d"%(np.sum(y_test==0)))
print("Ones: %d"%(np.sum(y_test==1)))



