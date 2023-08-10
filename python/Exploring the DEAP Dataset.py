import pandas as pd

# import and look at the type and shape of the participant_ratings file
participant_data = pd.read_csv("deap_data/metadata_csv/participant_ratings.csv")
participant_data

import numpy as np
a = participant_data[participant_data['Participant_id'] == 1]
# b = participant_data[participant_data['Trial'] == 1]
# c = pd.merge(a,b)
# type(np.int(c.Experiment_id)-1)
a

# DEAP preprocessed data construction
# Lets get a brief overview of one piece of data
# The data is cut up into 32 pieces each with their own dat file loadable via pickle

import cPickle
x = cPickle.load(open('deap_data/data_preprocessed_python/s01.dat', 'rb'))
print type(x)
print x['labels'].shape
print x['data'].shape

import cPickle
# Load the entire 32 patients accesible by number.
raw_data_dict = cPickle.load(open('deap_data/data_preprocessed_python/all_32.dat', 'rb'))
print type(raw_data_dict)

(raw_data_dict[1])

