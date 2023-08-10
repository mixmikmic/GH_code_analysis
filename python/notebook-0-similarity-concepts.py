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

# Create a dataframe manually to illustrate the examples
ratings = pd.DataFrame(columns = ["customer", "movie", "rating"], 
                       data=[
                           ['Ana','movie_1',1],
                           ['Ana', 'movie_2', 5],
                           ['Bob','movie_1',1],
                           ['Bob', 'movie_2', 5]])
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)

rating_1 = ratings_matrix.ix['Ana']
rating_2 = ratings_matrix.ix['Bob']

ratings_matrix

s_intersection =similarity.calculate_distance(rating_1, rating_2, 'intersection')
s_cosine = similarity.calculate_distance(rating_1, rating_2, 'cosine')
s_pearson = similarity.calculate_distance(rating_1, rating_2, 'pearson')
s_jaccard = similarity.calculate_distance(rating_1, rating_2, 'jaccard')

print("similarity intersection: ", s_intersection)
print("similarity cosine: ", s_cosine)
print("similarity pearson: ", s_pearson)
print("similarity jaccard: ", s_jaccard)

# Create a dataframe manually to illustrate the examples
ratings = pd.DataFrame(columns = ["customer", "movie", "rating"], 
                       data=[
                           ['Ana','movie_1',5],
                           ['Ana', 'movie_2', 1],
                           ['Bob','movie_1',1],
                           ['Bob', 'movie_2', 5]])
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)

rating_1 = ratings_matrix.ix['Ana']
rating_2 = ratings_matrix.ix['Bob']

ratings_matrix

s_intersection =similarity.calculate_distance(rating_1, rating_2, 'intersection')
s_cosine = similarity.calculate_distance(rating_1, rating_2, 'cosine')
s_pearson = similarity.calculate_distance(rating_1, rating_2, 'pearson')
s_jaccard = similarity.calculate_distance(rating_1, rating_2, 'jaccard')

print("similarity intersection: ", s_intersection)
print("similarity cosine: ", s_cosine)
print("similarity pearson: ", s_pearson)
print("similarity jaccard: ", s_jaccard)

# Create a dataframe manually to illustrate the examples
data=[['Ana','movie_1',5],['Ana', 'movie_2', 1],['Bob','movie_3',5],['Bob', 'movie_4', 5]]
ratings = pd.DataFrame(columns = ["customer", "movie", "rating"], data=data)
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)

rating_1 = ratings_matrix.ix['Ana']
rating_2 = ratings_matrix.ix['Bob']

ratings_matrix

s_intersection =similarity.calculate_distance(rating_1, rating_2, 'intersection')
s_cosine = similarity.calculate_distance(rating_1, rating_2, 'cosine')
s_pearson = similarity.calculate_distance(rating_1, rating_2, 'pearson')
s_jaccard = similarity.calculate_distance(rating_1, rating_2, 'jaccard')

print("similarity intersection: ", s_intersection)
print("similarity cosine: ", s_cosine)
print("similarity pearson: ", s_pearson)
print("similarity jaccard: ", s_jaccard)

# Create a dataframe manually to illustrate the examples
data=[['Ana','movie_1',5],['Ana', 'movie_2', 4],['Bob','movie_1',3],['Bob', 'movie_2', 2]]
ratings = pd.DataFrame(columns = ["customer", "movie", "rating"], data=data)
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)

rating_1 = ratings_matrix.ix['Ana']
rating_2 = ratings_matrix.ix['Bob']

ratings_matrix

s_intersection =similarity.calculate_distance(rating_1, rating_2, 'intersection')
s_cosine = similarity.calculate_distance(rating_1, rating_2, 'cosine')
s_pearson = similarity.calculate_distance(rating_1, rating_2, 'pearson')
s_jaccard = similarity.calculate_distance(rating_1, rating_2, 'jaccard')

print("similarity intersection: ", s_intersection)
print("similarity cosine: ", s_cosine)
print("similarity pearson: ", s_pearson)
print("similarity jaccard: ", s_jaccard)

# Create a dataframe manually to illustrate the examples
data=[['Ana','movie_1',5],['Ana', 'movie_2', 4],['Ana', 'movie_3', 4],['Bob','movie_1',3],['Bob', 'movie_2', 2]]
ratings = pd.DataFrame(columns = ["customer", "movie", "rating"], data=data)
ratings_matrix = ratings.pivot_table(index='customer', columns='movie', values='rating', fill_value=0)

rating_1 = ratings_matrix.ix['Ana']
rating_2 = ratings_matrix.ix['Bob']

ratings_matrix

s_intersection =similarity.calculate_distance(rating_1, rating_2, 'intersection')
s_cosine = similarity.calculate_distance(rating_1, rating_2, 'cosine')
s_pearson = similarity.calculate_distance(rating_1, rating_2, 'pearson')
s_jaccard = similarity.calculate_distance(rating_1, rating_2, 'jaccard')

print("similarity intersection: ", s_intersection)
print("similarity cosine: ", s_cosine)
print("similarity pearson: ", s_pearson)
print("similarity jaccard: ", s_jaccard)

