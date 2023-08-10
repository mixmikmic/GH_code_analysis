get_ipython().magic('matplotlib inline')
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from dataset_creation import create_data
create_data()

from models.base_convnet import basic_model

basic_model(lr=0.003, num_epochs=10)
basic_model(lr=0.001, num_epochs=2)

from models.inception_model import inception_model

inception_model(lr=0.0003, num_epochs=10)
inception_model(lr=0.0001, num_epochs=2)

from models.semantic_segmentation import semantic_segmentation

semantic_segmentation(lr=0.002, num_epochs=10)

from visualize_activations import visualize_activations

visualize_activations()

from plot_semantic_segmentation import create_ss_plots

create_ss_plots()

