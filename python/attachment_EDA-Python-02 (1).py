your_local_path="C:/sandbox/python/Datasets/"

import pandas as pd
import numpy as np

# Pass column names in names for each CSV

# Load the users data
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(your_local_path+'u.user', sep='|', names=u_cols,
                    encoding='latin-1')

# Load the ratings data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(your_local_path+'u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols

# Load the movies data
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv(your_local_path+'u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# Check the structure of Users data
users.shape

# Check few sample values of user data
users.head()

# Get information on user data
users.info()

# Get data type information of movies data
movies.dtypes

# Describe the spread of the users data
users.describe()

# Get sample movie ids
movies['movie_id'].head()

# Selecting a subset of columns from the movies data
movies[['movie_id','title']].head()

# Another example of subsetting (Putting a condition)
users[users.age<40].tail()

# More conditions
users[(users.age < 40) & (users.sex == 'F')].head(3)

# Create the index for the users dataset. Can run only once, do not run more than once, else you shall get an error.
users.set_index('user_id', inplace=False)
users.head()

# If you wish then please reset the index
users.reset_index(inplace=False)
users.head()

# Merge datasets - Movies, ratings & users
print(movies.shape)
print(movies.size)
movies.head()    # movie_id

ratings.head()
print(ratings.shape)
print(ratings.size)

users.head()
print(users.shape)
print(users.size)

# Merge datasets - Movies, ratings & users
movies.head()    # movie_id
ratings.head()   # movie_id & user_id
#users.head()     # user_id

movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)
lens

print(lens.head())
print(movies.shape)
movies.head()
print(ratings.shape)
print(users.shape)
lens.shape

# Let us answer some simple questions now. 
# Most rated movies
#print(lens.groupby('title').size())
most_rated = lens.groupby('title').size().sort_values(ascending=False)
most_rated.head(1)

# Another way to get the mentions. value_counts: The resulting object will be in descending order so that the first element is the most frequently-occurring 
lens.title.value_counts()[:20]

# Highest rated movies
highest_ratings = lens.groupby('title').agg({'rating':[np.size,np.mean,np.max,np.min]})
highest_ratings.head()

# Let us sort the output and see what we get
highest_ratings.sort_values([('rating', 'mean')], ascending=False).head()

# We shall consider movies that have been rated more than 100 times
atleast_100 = highest_ratings['rating']['size'] >= 100
highest_ratings[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot([10,20],[10,20],'ro')
plt.title("Distribution of users' ages") 
plt.ylabel('count of users')


ratings.rating.plot.hist(bins=5) 
plt.title("Ratings") 
plt.ylabel('movies count') 
plt.xlabel('rating') 
plt.axis([0,5,0,35000])
ratings.head(1)

# How to get the 397 th user's age
lens['age'][397]

# Let us create buckets age-wise
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)
lens.head(1)

labels = ['', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)
lens.head(1)

# Group these age buckets and check the rating behaviour. Check who has given max & min number of ratings
lens.groupby('occupation').agg({'rating': [np.size, np.mean]})

# Get the top 100 most mentions
most_100 = lens.groupby('movie_id').size().sort_values(ascending=False)[:100]

# Lets set the movie_id as the index
lens.set_index('movie_id', inplace=True)

# Lets split the observations by title and age group
by_age = lens.loc[most_100.index].groupby(['title', 'age_group'])
by_age.rating.mean().head(15)

# Make it more presentable using unstack
by_age.rating.mean().unstack(1).fillna(0)[10:20]

# Reset movie_id as index
lens.reset_index('movie_id', inplace=True)

# Let us pivot the data and split observations betweem male and female ratings
pivoted = lens.pivot_table(index=['movie_id', 'title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)
pivoted.head()

# Try to find the rating behaviour difference between male and female
pivoted['diff'] = pivoted.M - pivoted.F
pivoted.head()

# Reset movie_id as index again
pivoted.reset_index('movie_id', inplace=True)
pivoted

# Let us plot the rating differences between male and female and check for ourselves for patterns
disagreements = pivoted[pivoted.movie_id.isin(most_100.index)]['diff']
disagreements.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
plt.ylabel('Title')
plt.xlabel('Average Rating Difference');

