import pandas as pd
import numpy as np
from scipy.spatial import distance
from numpy.random import permutation
import math
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

nba = pd.read_csv("datasets/nba_2013.csv")

nba.head(10).transpose()

selected_player = nba[nba["player"] == "Manu Ginobili"]

selected_player

# Choose only the numeric columns (we'll use these to compute euclidean distance)

distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga',                     'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p',                     'x2pa', 'x2p.', 'efg.', 'ft', 'fta',                     'ft.', 'orb', 'drb', 'trb', 'ast', 'stl',                     'blk', 'tov', 'pf', 'pts']

np.array(distance_columns)

def euclidean_distance(row):
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)

# Find the distance from each player in the dataset to lebron.
player_distance = nba.apply(euclidean_distance, axis=1)

player_distance.head(10)

# Select only the numeric columns from the NBA dataset
nba_numeric = nba[distance_columns]

nba_numeric.head().transpose()

# Normalize all of the numeric columns

nba_normalized = (nba_numeric - nba_numeric.mean()) / nba_numeric.std()

nba_normalized.head()

# fill in NA values in nba_normalized
nba_normalized.fillna(0, inplace = True)

nba_normalized.head()

# find the normalized vector for Manu Ginobili

ginobili_normalized = nba_normalized[nba["player"] == "Manu Ginobili"]
ginobili_normalized.head()

# find the distance between ginobili and everyone else

euclidean_distances = nba_normalized.apply(lambda row: distance.euclidean(row, ginobili_normalized), axis = 1)

euclidean_distances.head()

distance_frame = pd.DataFrame(data = {"dist": euclidean_distances, "idx": euclidean_distances.index})

distance_frame.head()

distance_frame.sort_values("dist", inplace = True)

distance_frame.head()

# find the most similar player to ginobili
# the lowest distance to ginobili is ginobili
# the second lowest is the most similar to ginobili

second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_ginobili = nba.loc[int(second_smallest)]["player"]

most_similar_to_ginobili

# shuffle

nba.fillna(0, inplace = True)

random_indices = permutation(nba.index)

test_cutoff = math.floor(len(nba)/3)

test = nba.loc[random_indices[1:test_cutoff]]

train = nba.loc[random_indices[test_cutoff:]]

len(test)

len(train)

len(nba)

train.head()

test.head()

# columns that we'd be making predictions with. 

x_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.',              'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.',              'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb',              'ast', 'stl', 'blk', 'tov', 'pf']

# column that we want to predict.

y_column = ['pts']

# create the knn model
# look at the 5 closest neighbors

k = 5
knn = KNeighborsRegressor(n_neighbors = 5)
knn

# fit the model on the training data
knn.fit(train[x_columns], train[y_column])

# make point predictions on the test set using the fit model

predictions = knn.predict(test[x_columns])

predictions[0:5]

test.head()

# get actual values for the test set

actual = test[y_column]

# compute the MSE 

mse = (((predictions - actual) ** 2).sum()) / len(predictions)

mse

k = 6

neigh = NearestNeighbors(n_neighbors = k)
neigh.fit(nba_normalized)
neighbors = neigh.kneighbors(ginobili_normalized, return_distance = False)
neighbors = neighbors.tolist()[0]

res = pd.DataFrame(nba.iloc[neighbors, :])
res.transpose()

