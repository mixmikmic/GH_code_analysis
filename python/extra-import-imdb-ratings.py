import os
os.chdir('..')

import pandas as pd
import os 
import src.utils as utils

# Enable logging on Jupyter notebook
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# paths
dataset_folder = os.getcwd() + '/data/'
imdb_exported_ratings_file = dataset_folder +'/ratings.csv'
links_file = dataset_folder + '/ml-latest-small/links.csv'
ratings_file = dataset_folder + '/ratings-imdb.csv'

utils.import_imdb_ratings(imdb_exported_ratings_file, links_file, ratings_file)

