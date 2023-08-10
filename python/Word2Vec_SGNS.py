get_ipython().magic('matplotlib inline')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from gensim.models import Word2Vec

# import our own made functions
from load import *
from output import *

# create two model from two different periods by loading the pickle files 
# that were created after cleaning the data.

# note that there are much more articles in recent years so the periods
# must not necessarily be of the same length.
data_old = loadYears("../data/Cleaned/GDL", range(1798,1860))
data_new = loadYears("../data/Cleaned/GDL", range(1950,1960))

model_old = createModel(data_old)
model_new = createModel(data_new)

# now, we create the transformation matrix using Procrustes, and apply it
# to the earlier period. We then return the modified first model (with the
# transformation applied) and the new model.

modZ, modB = createTransformationMatrix(model_old,model_new)

# we can call this function to compare the shift of a selected word, with a
# certain number of neighbour for each period to get an idea of the related
# words and context
# t-SNE is used to show the multidimentional vectors of 300 features
# into a 2 dimentional space.

# the red dots are from the earlier datasets and the blue dots from the 
# most recent one. The arrow shows the shift estimated using t-SNE.

# Note: t-SNE is stochastic, and thus it might give a different result at
#   each trial. Just run it again if the result is not satisfactory.

visualizeWord(modZ, modB, 'bâtiment', 4)

# TODO: here insert just a couple of interesting words
#    some ideas: armée, transport, ...
visualizeWord(modZ, modB, 'vapeur', 4) 



