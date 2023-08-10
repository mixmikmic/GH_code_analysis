import logging
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

import numpy
from keras.models import Sequential
from keras.layers import Dense
from utils import keras_backend_utils as kbu

seed = 7
numpy.random.seed(seed)

# do some work with Theano, 
from keras import backend as K
logging.info("Current backend : {}".format(K._BACKEND))
logging.info("Toggling the backend ...")

#then toggle to using Tensorflow as a backend
kbu.toggle_keras_backend()
logging.info("Current backend : {}".format(K._BACKEND))

# Toggle back to Theano
logging.info("Toggling the backend the 2nd time ...")
kbu.toggle_keras_backend()
logging.info("Current backend : {}".format(K._BACKEND))



