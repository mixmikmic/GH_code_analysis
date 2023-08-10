import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().magic('matplotlib inline')

board_games = pd.read_csv("board_games.csv")

print board_games.head(10)

# Check the column names
print board_games.columns

# Data types of columns

print board_games.dtypes

#Is there any missing values?
board_games.isnull().sum()

# Drop all missing values
board_games = board_games.dropna(axis = 0)

board_games.isnull().sum()

# See if there's any games with no reviewers
has_reviewers = board_games["users_rated"]!=0
print sum(has_reviewers)
print board_games.shape

# Pick only the rows that has any reviewers
board_games = board_games[has_reviewers]

board_games.shape

board_games["playingtime"].describe()

#which game has the highest playing time?

print board_games.loc[board_games["playingtime"].idxmax()]

# it's kind of funny that only one person played the game with the highest playing time.

plt.hist(board_games["average_rating"])
plt.xlabel("distribution of the average rating of the board games")

board_games["average_rating"].describe()

# Which games had rating above 8? Might be a good place to find recommendations



board_games[board_games["average_rating"]>8]

board_games[board_games["average_rating"]>8].describe()

# How many types of board games are here? 

print board_games["type"].value_counts()

# What's the distribution of the player counts?

board_games["maxplayers"].value_counts().sort_index()

# Note : There's no way 3173 games have a max player count of 0.0. Very very weird!

# Drop all non-numeric columns 

numeric = board_games.iloc[:,3:]
game_mean = numeric.apply(np.mean,axis=1)
game_std = numeric.apply(np.std, axis =1)

print numeric.shape
print board_games.shape

print numeric.columns

# practicing clustering based on higher values and variance of features vs lower values and variance of features here.

from sklearn.cluster import KMeans

Cluster = KMeans(n_clusters = 5)
Cluster.fit(numeric)

# Find the labels of the cluster
labels = Cluster.labels_

plt.scatter(x=game_mean,y=game_std,c=labels)

# let's check the correlations 

correlations = numeric.corr()

correlations

print correlations["average_rating"]

# Let's do the linear regression this time

columns = list(board_games.columns)
print columns
remove_list = ['id','type','name','average_rating','bayes_average_rating']
for x in remove_list:
    columns.remove(x)
print columns

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(board_games[columns],board_games['average_rating'])
predictions = reg.predict(board_games[columns])

mse = np.mean((predictions - board_games['average_rating'])**2)

print mse
print mse**(0.5)



