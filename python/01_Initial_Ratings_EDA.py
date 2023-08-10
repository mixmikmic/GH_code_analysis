# <!-- collapse=True -->
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from IPython.display import Markdown
from os.path import join
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use modified seaborn-darkgrid style with bigger text
plt.style.use('../big-darkgrid.mplstyle')

# <!-- collapse=True -->
ratings_feather = '../preprocessed/ratings.feather'
ratings_csv = '../raw/ratings.csv'
if exists(ratings_feather):
    # Reading in feather file saves time
    ratings_df = pd.read_feather(ratings_feather)
else:
    ratings_df = pd.read_csv('../raw/ratings.csv')
    ratings_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    ratings_df['timestamp'] = ratings_df['timestamp'].apply(datetime.fromtimestamp)
    ratings_df['year'] = ratings_df['timestamp'].dt.year
    ratings_df['month'] = ratings_df['timestamp'].dt.month
    ratings_df['day'] = ratings_df['timestamp'].dt.day
    ratings_df['hour'] = ratings_df['timestamp'].dt.hour
    ratings_df['minute'] = ratings_df['timestamp'].dt.minute
    # Save to feather file for faster loading later
    ratings_df.to_feather(ratings_feather)
ratings_df.head()

# <!-- collapse=True -->
ax = sns.countplot(x='rating', data=ratings_df)
ax.set_title('Rating Frequencies')
plt.show()

# <!-- collapse=True -->
ratings_pivot = pd.pivot_table(ratings_df, values='timestamp', index='rating', columns='year', aggfunc='count')
ratings_pivot = ratings_pivot.apply(lambda x: x/np.sum(x), axis=0)
ratings_pivot = ratings_pivot.sort_index(ascending=False)
ratings_pivot

# <!-- collapse=True -->
fig, ax = plt.subplots(figsize=(15, 5))
sns.heatmap(ratings_pivot, ax=ax)
ax.set_title('Rating Density by Year\n(Sum over each year is 1)')
plt.show()

switch_timestamp = ratings_df[ratings_df['rating'].isin([0.5, 1.5, 2.5, 3.5, 4.5])].iloc[0]['timestamp']
ratings_df[ratings_df['timestamp'] == switch_timestamp]

# <!-- collapse=True -->
sns.countplot(x='rating', data=ratings_df[ratings_df['timestamp']<switch_timestamp])
plt.title('Rating Frequencies Before 2/18/2003')
plt.tight_layout()
plt.show()
sns.countplot(x='rating', data=ratings_df[ratings_df['timestamp']>=switch_timestamp])
plt.title('Rating Frequencies After 2/18/2003')
plt.tight_layout()
plt.show()

# <!-- collapse=True -->
year_counts = ratings_df['year'].value_counts().sort_index()
plt.plot(year_counts.index, year_counts, 'o-')
plt.xlabel('Year')
plt.ylabel('# of Ratings')
plt.title('# of Ratings per Year')
plt.show()
n_1995 = np.sum(ratings_df['year']==1995)
display(Markdown("Here are the {} ratings from 1995:".format(n_1995)))
display(ratings_df[ratings_df['year']==1995])

# <!-- collapse=True -->
means_stds_by_year = ratings_df.groupby('year')['rating'].agg(['mean', 'std']).reset_index()
years, means, stds = means_stds_by_year['year'], means_stds_by_year['mean'], means_stds_by_year['std']
l, = plt.plot(years, means, 'o-', label='Mean Rating')
plt.fill_between(years, means-stds, means+stds, alpha=0.3, color=l.get_color())
plt.ylim([0, 5])
plt.xlabel('Year')
plt.ylabel('Rating')
plt.title(r'Rating Means $\pm$ Std vs Year')
plt.tight_layout()
plt.show()

# <!-- collapse=True -->
n_users = []
n_movies = []
years = np.unique(ratings_df['year'])
for year in years:
    n_users.append(ratings_df[ratings_df['year'] <= year]['userId'].nunique())
    n_movies.append(ratings_df[ratings_df['year'] <= year]['movieId'].nunique())
plt.plot(years, n_users, 'o-')
plt.xlabel('Year')
plt.ylabel('# of Users')
plt.tight_layout()
plt.show()
plt.plot(years, n_movies, 'o-')
plt.xlabel('Year')
plt.ylabel('# of Movies')
plt.tight_layout()
plt.show()

# <!-- collapse=True -->
user_counts = ratings_df['userId'].value_counts()
movie_counts = ratings_df['movieId'].value_counts()
plt.plot(user_counts.values)
plt.xlabel('User')
plt.ylabel('# Movies Rated')
plt.tight_layout()
plt.show()
plt.plot(movie_counts.values)
plt.xlabel('Movie')
plt.ylabel('# Users Rating')
plt.tight_layout()
plt.show()

# <!-- collapse=True -->
highest_num_movies_rated = user_counts.iloc[0]
highest_num_ratings_on_movie = movie_counts.iloc[0]
percent_movies_with_1_rating = 100 * (movie_counts==1).sum()/len(movie_counts)
Markdown("""
You'll see that there are a very small number of highly
active users (one who rated {} movies!), and a few
movies with huge numbers of ratings, the highest of which
is [Pulp Fiction](https://movielens.org/movies/296) coming
in with  a whopping {} ratings in this dataset (90,505
in total as of 3/12/2018)! This quickly drops off to just
a few ratings. Users were only included in this dataset if
they had $\ge$ 20 ratings, so 20 is the minimum that the
user curve drops off to, but there was no such limit for movies.
In fact, about {}% of the movies in this dataset had exactly 1 rating.
""".format(
        highest_num_movies_rated,
        highest_num_ratings_on_movie,
        int(percent_movies_with_1_rating + 0.5),
    )
)

# <!-- collapse=True -->
ax = sns.distplot(user_counts)
ax.set_xlabel('# of Movies Rated')
ax.set_ylabel(r'$PDF$')
ax.set_ylim([-0.0002, 0.0047])
ax.set_title('User Distribution')
plt.show()
ax = sns.distplot(movie_counts)
ax.set_xlabel('# of Users Rating')
ax.set_ylabel(r'$PDF$')
ax.set_ylim([-0.00002, 0.00075])
ax.set_title('Movie Distribution')
plt.show()

# <!-- collapse=True -->
log_user_counts = np.log10(user_counts)
ax = sns.distplot(log_user_counts)
ax.set_xlabel(r'$\log_{10}($# of Movies Rated$)$')
ax.set_ylabel(r'$PDF$')
plt.show()
log_movie_counts = np.log10(movie_counts)
ax = sns.distplot(log_movie_counts)
ax.set_xlabel(r'$\log_{10}($# of Users Rating$)$')
ax.set_ylabel(r'$PDF$')
plt.show()

# <!-- collapse=True -->
num_users = ratings_df['userId'].nunique()
num_movies = ratings_df['movieId'].nunique()
num_ratings_given = len(ratings_df)
user_movies_matrix_density = num_ratings_given / num_users / num_movies
Markdown("""
We have about {:,} users and about {:,} movies, giving a possible total of about
{:4.2f} **billion** ratings, but we only have about {:4.1f} million ratings,
giving a matrix density of only {:7.5f}!
This is a very sparse matrix. This means that of every 1000 possible
user-movie combos that could have happened, only about {} actually did.
""".format(
        num_users, num_movies, (num_users * num_movies)/1e9, num_ratings_given/1e6,
        user_movies_matrix_density, int(user_movies_matrix_density*1000)
    )
)

