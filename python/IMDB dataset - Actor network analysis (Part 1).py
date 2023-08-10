# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the csv file
movies = pd.read_csv("movie_metadata.csv")

# Print the column names
movies.columns

# How many movies do we have?
movies.shape

# Drop the duplicates, just in case.
movies = movies.drop_duplicates()
movies.shape

# Summary statistics of num_voted_users
movies['num_voted_users'].describe()

plt.hist(movies['num_voted_users'], 'auto', alpha=0.75)
plt.show()

voted_movies = movies[movies['num_voted_users'] >= 34000]

# Summary statistics of the score:
voted_movies['imdb_score'].describe()

plt.hist(voted_movies['imdb_score'], 'auto', normed = 100, alpha=0.75)
plt.show()

high_score = voted_movies[(voted_movies['imdb_score'] >= 7.5)]
low_score = voted_movies[(voted_movies['imdb_score'] <= 6.2)]

# Sort the dataset by score:
high_score = high_score.sort_values(by = "imdb_score", ascending = False)

high_score

# Only extract the info we want:
high_score_df = high_score[['movie_title','actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score']]
high_score_df

# Sort by scores:
low_score = low_score.sort_values(by = "imdb_score", ascending = True)
# Extract info needed:
low_score_df = low_score[['movie_title','actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score']]
low_score_df

def gephi(df):
    df = df.drop(['movie_title', 'imdb_score'], axis = 1) # drop extra columns
    df = df.replace(' ', '_', regex=True) # replace whitespace by underscore, since Gephi doesn't recognize spaces
    df = df.reset_index(drop=True) # reset indices
    return(df)

high_score_df = gephi(high_score_df)
high_score_df

low_score_df = gephi(low_score_df)
low_score_df

high_score_df.to_csv("high_score_df.csv", sep = ";", index = False, header = False)
low_score_df.to_csv("low_score_df.csv", sep = ";", index = False, header = False)

