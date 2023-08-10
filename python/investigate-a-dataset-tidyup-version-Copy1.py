import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# Stop floats from displaying as scientific notation
pd.options.display.float_format = '{:20,.2f}'.format

# Load your data and print out a few lines. Perform operations to inspect data
# types and look for instances of missing or possibly errant data.
movies = pd.read_csv('tmdb-movies.csv')

movies.head()

movies.info()

movies.nunique()

null_check = movies.loc[:, ['id', 'budget_adj']].sort_values(by=['budget_adj'], ascending=True)
null_check.head()

print("I am imdb_id: ", type(movies['imdb_id'][0]))
print("I am original_title: ", type(movies['original_title'][0]))
print("I am cast: ", type(movies['cast'][0]))
print("I am homepage: ", type(movies['homepage'][0]))
print("I am director: ", type(movies['director'][0]))
print("I am tagline: ", type(movies['tagline'][0]))
print("I am keywords: ", type(movies['keywords'][0]))
print("I am overview: ", type(movies['overview'][0]))
print("I am genres: ", type(movies['genres'][0]))
print("I am production_companies: ", type(movies['production_companies'][0]))
print("I am release_date: ", type(movies['release_date'][0]))

# create an extra column and mark a row as True where a duplicate itle is found
movies['is_duplicate_title'] = movies.duplicated(['original_title'])

# filter anything that is True
movies_dupe_title_filter = movies[movies['is_duplicate_title'] == True]

movies_dupe_title_filter

# use this cell to spot check titles for differences
movies_title_check = movies[movies['original_title'] == 'Robin Hood']
movies_title_check.head()

movies['is_duplicate_id'] = movies.duplicated(['id'])

movies_dupe_id_filter = movies[movies['is_duplicate_id'] == True]

movies_dupe_id_filter.head()

movies_id_check_dupe = movies[movies['id'] == 42194]
movies_id_check_dupe.head()

movies.drop_duplicates(subset=['id'],inplace=True)

# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.

movies.drop(['imdb_id', 'budget', 'revenue', 'homepage', 'tagline', 'overview', 'keywords', 'is_duplicate_title', 'is_duplicate_id'], axis=1, inplace=True)

movies.head(1)

movies['release_date'] = pd.to_datetime(movies['release_date'])

# check it's worked
type(movies['release_date'][0])

# prepare dataframe for question 1
movie_genres = movies.copy()
movie_genres.drop(['original_title', 'cast', 'director', 'runtime', 'release_date', 'production_companies', 'vote_count', 'vote_average','budget_adj', 'revenue_adj'], axis=1, inplace=True)
movie_genres.head(1)

# drop NaN values - only targets genres at this stage
movie_genres.dropna(axis=0, how='any', inplace=True)

# grab the id and genres column
genre = movie_genres.loc[:, ['id', 'genres']]
genre.head(5)

# split the genres cells by the pipe and add to a list
genre_list = genre['genres'].str.split('|').tolist()
genre_list[:5]

# loop through each iterable (a nested list) in genre_list
# check that each element is indeed a list
# convert to list if not

for i in range(len(genre_list)):
    if not isinstance(genre_list[i], list):
        genre_list[i] = [genre_list[i]]

        
# an error occured when cell below was first run, hence loop

"""
Create a new dataframe using genre_list and the id column 
of the 'genre' dataframe as the index. As this will result in
multiple columns with an individual genre value per id, we need
to apply .stack() to pivot the data:
https://pandas.pydata.org/pandas-docs/stable/reshaping.html
"""
stacked_genre = pd.DataFrame(genre_list, index=genre['id']).stack()

print(stacked_genre.head())

# needs comment
stacked_genre = stacked_genre.reset_index()

print(stacked_genre.head())

stacked_genre = stacked_genre.loc[:, ['id', 0]]

print(stacked_genre.head())

stacked_genre.columns = ['id', 'genre']

print(stacked_genre.head())

merged = pd.merge(movie_genres, stacked_genre, on='id', how='left')
merged.drop(['genres'], axis=1, inplace=True)

merged.head()

merged['genre'].value_counts().plot(kind='bar', figsize=(16, 8));

merged.drop(['id'], axis=1, inplace=True)
merged_grouped = merged.groupby(['release_year', 'genre']).mean()

merged_grouped.head(20)

# drama
drama = merged[merged['genre'] == 'Drama'].copy()
drama.drop(['genre'], axis=1, inplace=True)
drama_popularity = drama.groupby(['release_year']).mean().reset_index()
drama_popularity.rename(columns={'popularity':'popularity_drama'}, inplace=True)

# comedy
comedy = merged[merged['genre'] == 'Comedy'].copy()
comedy.drop(['genre'], axis=1, inplace=True)
comedy_popularity = comedy.groupby(['release_year']).mean().reset_index()
comedy_popularity.rename(columns={'popularity':'popularity_comedy'}, inplace=True)

# thriller
thriller = merged[merged['genre'] == 'Thriller'].copy()
thriller.drop(['genre'], axis=1, inplace=True)
thriller_popularity = thriller.groupby(['release_year']).mean().reset_index()
thriller_popularity.rename(columns={'popularity':'popularity_thriller'}, inplace=True)

# action
action = merged[merged['genre'] == 'Action'].copy()
action.drop(['genre'], axis=1, inplace=True)
action_popularity = action.groupby(['release_year']).mean().reset_index()
action_popularity.rename(columns={'popularity':'popularity_action'}, inplace=True)

# romance
romance = merged[merged['genre'] == 'Romance'].copy()
romance.drop(['genre'], axis=1, inplace=True)
romance_popularity = romance.groupby(['release_year']).mean().reset_index()
romance_popularity.rename(columns={'popularity':'popularity_romance'}, inplace=True)
romance_popularity.head()

# horror
horror = merged[merged['genre'] == 'Horror'].copy()
horror.drop(['genre'], axis=1, inplace=True)
horror_popularity = horror.groupby(['release_year']).mean().reset_index()
horror_popularity.rename(columns={'popularity':'popularity_horror'}, inplace=True)

# adventure
adventure = merged[merged['genre'] == 'Adventure'].copy()
adventure.drop(['genre'], axis=1, inplace=True)
adventure_popularity = adventure.groupby(['release_year']).mean().reset_index()
adventure_popularity.rename(columns={'popularity':'popularity_adventure'}, inplace=True)

# crime
crime = merged[merged['genre'] == 'Crime'].copy()
crime.drop(['genre'], axis=1, inplace=True)
crime_popularity = romance.groupby(['release_year']).mean().reset_index()
crime_popularity.rename(columns={'popularity':'popularity_crime'}, inplace=True)

# family
family = merged[merged['genre'] == 'Family'].copy()
family.drop(['genre'], axis=1, inplace=True)
family_popularity = romance.groupby(['release_year']).mean().reset_index()
family_popularity.rename(columns={'popularity':'popularity_family'}, inplace=True)

# science fiction
scifi = merged[merged['genre'] == 'Science Fiction'].copy()
scifi.drop(['genre'], axis=1, inplace=True)
scifi_popularity = romance.groupby(['release_year']).mean().reset_index()
scifi_popularity.rename(columns={'popularity':'popularity_scifi'}, inplace=True)

genre_merge = pd.merge(drama_popularity, comedy_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, thriller_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, action_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, romance_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, horror_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, adventure_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, crime_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, family_popularity, on='release_year', how='left')
genre_merge = pd.merge(genre_merge, scifi_popularity, on='release_year', how='left')

fig, ax = plt.subplots(figsize=(16, 8))
plt.title('Average popularity of films by genre')
plt.ylabel('Average Annual Popularity')
plt.xlabel('Release Year')
plt.xticks(np.arange(1960, 2016, 2))
ax.plot('release_year', 'popularity_drama', data=genre_merge, label="Drama")
ax.plot('release_year', 'popularity_comedy', data=genre_merge, label="Comedy")
ax.plot('release_year', 'popularity_thriller', data=genre_merge, label="Thriller")
ax.plot('release_year', 'popularity_action', data=genre_merge, label="Action")
ax.plot('release_year', 'popularity_romance', data=genre_merge, label="Romance")
ax.plot('release_year', 'popularity_horror', data=genre_merge, label="Horror")
ax.plot('release_year', 'popularity_adventure', data=genre_merge, label="Adventure")
ax.plot('release_year', 'popularity_crime', data=genre_merge, label="Crime")
ax.plot('release_year', 'popularity_family', data=genre_merge, label="Family")
ax.plot('release_year', 'popularity_scifi', data=genre_merge, label="Sci-Fi")
ax.legend(loc='upper left');

# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.

# df_genres_test = df_genres[df_genres.release_year != 2015]
# use this later for dropping specific 0 values from budget_adj



