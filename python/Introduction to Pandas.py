# Note: Adjust the name of the folder to match your local directory

get_ipython().system('ls ./movielens')

get_ipython().system('ls ./movielens')

get_ipython().system('cat ./movielens/movies.csv | wc -l')

get_ipython().system('head -5 ./movielens/ratings.csv')

movies = pd.read_csv('./movielens/movies.csv', sep=',')
print(type(movies))
movies.head(15)

# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970

tags = pd.read_csv('./movielens/tags.csv', sep=',')
tags.head()

ratings = pd.read_csv('./movielens/ratings.csv', sep=',', parse_dates=['timestamp'])
ratings.head()

# For current analysis, we will remove timestamp (we will come back to it!)

del ratings['timestamp']
del tags['timestamp']

#Extract 0th row: notice that it is infact a Series

row_0 = tags.iloc[0]
type(row_0)

print(row_0)

row_0.index

row_0['userId']

'rating' in row_0

row_0.name

row_0 = row_0.rename('first_row')
row_0.name

tags.head()

tags.index

tags.columns

# Extract row 0, 11, 2000 from DataFrame

tags.iloc[ [0,11,2000] ]

ratings['rating'].describe()

ratings.describe()

ratings['rating'].mean()

ratings.mean()

ratings['rating'].min()

ratings['rating'].max()

ratings['rating'].std()

ratings['rating'].mode()

ratings.corr()

filter_1 = ratings['rating'] > 5
print(filter_1)
filter_1.any()

filter_2 = ratings['rating'] > 0
filter_2.all()

movies.shape

#is any row NULL ?

movies.isnull().any()

ratings.shape

#is any row NULL ?

ratings.isnull().any()

tags.shape

#is any row NULL ?

tags.isnull().any()

tags = tags.dropna()

#Check again: is any row NULL ?

tags.isnull().any()

tags.shape

get_ipython().magic('matplotlib inline')

ratings.hist(column='rating', figsize=(15,10))

ratings.boxplot(column='rating', figsize=(15,20))

tags['tag'].head()

movies[['title','genres']].head()

ratings[-10:]

tag_counts = tags['tag'].value_counts()
tag_counts[-10:]

tag_counts[:10].plot(kind='bar', figsize=(15,10))

is_highly_rated = ratings['rating'] >= 4.0

ratings[is_highly_rated][30:50]

is_animation = movies['genres'].str.contains('Animation')

movies[is_animation][5:15]

movies[is_animation].head(15)

ratings_count = ratings[['movieId','rating']].groupby('rating').count()
ratings_count

average_rating = ratings[['movieId','rating']].groupby('movieId').mean()
average_rating.head()

movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.head()

movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.tail()

tags.head()

movies.head()

t = movies.merge(tags, on='movieId', how='inner')
t.head()

avg_ratings = ratings.groupby('movieId', as_index=False).mean()
del avg_ratings['userId']
avg_ratings.head()

box_office = movies.merge(avg_ratings, on='movieId', how='inner')
box_office.tail()

is_highly_rated = box_office['rating'] >= 4.0

box_office[is_highly_rated][-5:]

is_comedy = box_office['genres'].str.contains('Comedy')

box_office[is_comedy][:5]

box_office[is_comedy & is_highly_rated][-5:]

movies.head()

movie_genres = movies['genres'].str.split('|', expand=True)

movie_genres[:10]

movie_genres['isComedy'] = movies['genres'].str.contains('Comedy')

movie_genres[:10]

movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)

movies.tail()

tags = pd.read_csv('./movielens/tags.csv', sep=',')

tags.dtypes

tags.head(5)

tags['parsed_time'] = pd.to_datetime(tags['timestamp'], unit='s')


tags['parsed_time'].dtype

tags.head(2)

greater_than_t = tags['parsed_time'] > '2015-02-01'

selected_rows = tags[greater_than_t]

tags.shape, selected_rows.shape

tags.sort_values(by='parsed_time', ascending=True)[:10]

average_rating = ratings[['movieId','rating']].groupby('movieId', as_index=False).mean()
average_rating.tail()

joined = movies.merge(average_rating, on='movieId', how='inner')
joined.head()
joined.corr()

yearly_average = joined[['year','rating']].groupby('year', as_index=False).mean()
yearly_average[:10]

yearly_average[-20:].plot(x='year', y='rating', figsize=(15,10), grid=True)

