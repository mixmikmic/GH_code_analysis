import pandas as pd

import plotly.offline as offline
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()

offline.init_notebook_mode()

## Read ratings data data
ratings_filename = "../data/ratings.csv"

ratings_df = pd.read_csv(ratings_filename)
print(ratings_df.shape)
count_ratings = ratings_df.shape[0]
ratings_df.head(10)

## get a few because why not
ratings_df = ratings_df.sample(10000)

## group by userId and look at the top rated movie for each user
ratings_df = ratings_df.sort_values('rating', ascending = False)
ratings_userid_grouped = ratings_df.groupby(['userId'])

ratings_userid_grouped.head(1) #.head(5)

## get some basic stats on each movie's ratings
ratings_movideid_grouped = ratings_df.groupby('movieId')
## focus only on 'rating'
# ratings_movideid_grouped[['rating']].describe().head(10)

## get the mean rating for different movies by user id
## there is two ways --

## 1
ratings_df.groupby([ratings_df['movieId'],ratings_df['userId']]).mean()

## 2
ratings_df['rating'].groupby([ratings_df['movieId'],ratings_df['userId']]).mean().head(5)

ratings_movideid_grouped.head(1)

## find if a user's rating is higher or lower than the mean rating given to the movie
## define the tansformation
normalise_rating = lambda x: x/x.mean()
## transform the rating
normal_ratings_df = ratings_movideid_grouped[['rating']].transform(normalise_rating)

## merge it into the orgiginal data
ratings_df.merge(normal_ratings_df, left_index = True, right_index = True).head(5)

