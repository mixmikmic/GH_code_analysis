from sklearn.datasets import load_boston
import pandas as pd, numpy as np
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import seaborn  as sns 
from sklearn.datasets import load_boston
import pylab as pl
import matplotlib.pyplot as plt

movies = pd.read_csv('/Users/GGV/Desktop/sell_Data_Analyst/movies.csv',encoding = "ISO-8859-1")
ratings = pd.read_csv('/Users/GGV/Desktop/sell_Data_Analyst/ratings.csv')
users = pd.read_csv('/Users/GGV/Desktop/sell_Data_Analyst/users.csv')

movies.head(2)

ratings.head(2)

users.head(2)

ratings.info()

ratings.mean()

ratings.describe()

ratings.describe()

users.info()

users.describe()

movies.columns

movies.groupby('Action').count()['movie_id']

movies.groupby('Horror').count()['movie_id']

movies[['Action', 'Horror']].head()

movies[(movies['Action']== 1)& (movies['Horror']== 1)]['title']

pd.DataFrame(movies.groupby(['Action', 'Horror']).count()['movie_id'])

ratings.mean()

len(ratings[ratings['rating'] > ratings.rating.mean()])

# dataframe merge 

final_table = pd.DataFrame()
final_table = pd.merge(users, ratings, on='user_id')
final_table = pd.merge(final_table,movies, on = 'movie_id' )

final_table.head(2)

final_table.groupby("gender").agg({"Action": np.sum,"Adventure":np.sum,"Animation":np.sum,"Comedy":np.sum,"Crime":np.sum,"Documentary":np.sum,"Drama":np.sum,"Fantasy":np.sum,"Film-Noir":np.sum,"Horror": np.sum,"Musical": np.sum,"Mystery": np.sum,"Romance": np.sum,"Sci-Fi": np.sum,"Thriller": np.sum,"War": np.sum,"Western": np.sum})

# helper function 

def round(x):
    return int(round(x/5.0)*5.0)

users['age_'] = (users['age']/5.0).astype(int)*5.0


users_ = pd.DataFrame()
users_ = pd.merge(users, ratings, on='user_id')

pd.DataFrame(users_.groupby(['age_','rating']).count()['user_id'])

# helper function 

import datetime
def unix_datetime(x):
    temp = datetime.datetime.fromtimestamp(x).strftime('%Y-%m')
    #temp = datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
    return temp

# unix time -> timestamp 

ratings['datetime'] = ratings['timestamp'].apply(lambda x :unix_datetime(x) )

# period confine 

rating_filter = ratings[(ratings['datetime']>'1997-10') & (ratings['datetime']<'1998-03')].sort(['datetime','movie_id'])


rating_filter.head()

rating_filter_ = rating_filter.groupby(['datetime','movie_id']).count().reset_index()

rating_filter_


rating_filter_.sort(['datetime','user_id'],ascending=False).groupby('datetime').head(1).sort('datetime')

# cohort analysis PANDAS : http://www.gregreda.com/2015/08/23/cohort-analysis-with-python/

