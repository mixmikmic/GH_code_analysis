import os
os.chdir('..')

# Import all the packages we need to generate recommendations
import pandas as pd
import src.utils as utils
import src.recommenders as recommenders
import src.similarity as similarity

# Enable logging on Jupyter notebook
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# downloads and unzips dataset from MovieLens 
#'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
dataset_folder = os.getcwd()+'/data/'
dataset_folder_ready = utils.load_dataset(dataset_folder)

# Export IMDB ratings to the right format 
imdb_ratings = dataset_folder +'/ratings.csv'
links_file = dataset_folder + '/ml-latest-small/links.csv'
ratings_file = dataset_folder + '/ratings-imdb.csv'
utils.import_imdb_ratings(imdb_ratings, links_file, ratings_file)

# adds personal ratings to original dataset ratings file.
[ratings, my_customer_number] = utils.merge_datasets(dataset_folder_ready, ratings_file)

# the data is stored in a long pandas dataframe
# we need to pivot the data to create a [user x movie] matrix
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)
ratings_matrix = ratings_matrix.transpose()

# the personal ratings are now stored together with the rest of the ratings
ratings.ix[ratings.customer == my_customer_number]

# A list with some of the movies in the dataset
movie_list = pd.DataFrame(ratings_matrix.index)
movie_list.head(20)

