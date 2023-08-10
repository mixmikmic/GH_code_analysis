import os
os.chdir('..')

# Import all the packages we need to generate recommendations
import numpy as np
import pandas as pd
import src.utils as utils
import src.recommenders as recommenders
import src.similarity as similarity

# Enable logging on Jupyter notebook
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# imports necesary for plotting
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# loads the dataset (assumes the data is downloaded)
dataset_folder = os.getcwd()+'/data/'
imdb_ratings = dataset_folder +'/ratings.csv'
links_file = dataset_folder + '/ml-latest-small/links.csv'
ratings_file = dataset_folder + '/ratings-imdb.csv'

# adds personal ratings to original dataset ratings file.
dataset_folder_ready = utils.load_dataset(dataset_folder)
utils.import_imdb_ratings(imdb_ratings, links_file, ratings_file)
[ratings, my_customer_number] = utils.merge_datasets(dataset_folder_ready, ratings_file)

# the data is stored in a long pandas dataframe
# we need to pivot the data to create a [user x movie] matrix
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)
ratings_matrix = ratings_matrix.transpose()

# find similar movies 
# try with different movie titles and see what happens 
movie_title = 'Star Wars: Episode VI - Return of the Jedi (1983)'
similarity_type = "cosine"
logger.info('top-10 movies similar to %s, using %s similarity', movie_title, similarity_type)
print(similarity.compute_nearest_neighbours(movie_title, ratings_matrix, similarity_type)[0:10])

# find similar movies 
# try with different movie titles and see what happens 
movie_title = 'All About My Mother (Todo sobre mi madre) (1999)'
similarity_type = "pearson"
logger.info('top-10 movies similar to: %s, using %s similarity', movie_title, similarity_type)
print(similarity.compute_nearest_neighbours(movie_title, ratings_matrix, similarity_type)[0:10])

# get recommendations for a single user
recommendations = recommenders.recommend_uknn(ratings, my_customer_number, K=200, similarity_metric='cosine', N=10)
recommendations

# get recommendations for a single user
recommendations = recommenders.recommend_iknn(ratings, my_customer_number, K=100, similarity_metric='cosine')
recommendations



