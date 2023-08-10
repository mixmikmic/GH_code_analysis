import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
# Stop floats from displaying as scientific notation
pd.options.display.float_format = '{:20,.2f}'.format

# Load your data and print out a few lines. Perform operations to inspect data
# types and look for instances of missing or possibly errant data.
df = pd.read_csv('tmdb-movies.csv')

df.head()

df.info()

df.sort_values(by=['vote_count'], ascending=False)

df.nunique()

print("I am imdb_id: ", type(df['imdb_id'][0]))
print("I am original_title: ", type(df['original_title'][0]))
print("I am cast: ", type(df['cast'][0]))
print("I am homepage: ", type(df['homepage'][0]))
print("I am director: ", type(df['director'][0]))
print("I am tagline: ", type(df['tagline'][0]))
print("I am keywords: ", type(df['keywords'][0]))
print("I am overview: ", type(df['overview'][0]))
print("I am genres: ", type(df['genres'][0]))
print("I am production_companies: ", type(df['production_companies'][0]))
print("I am release_date: ", type(df['release_date'][0]))

# create an extra column and mark a row as True where a duplicate itle is found
df['is_duplicate_title'] = df.duplicated(['original_title'])

# filter anything that is True
df_dupe_title_filter = df[df['is_duplicate_title'] == True]

df_dupe_title_filter

# use this cell to spot check titles for differences
df_2 = df[df['original_title'] == 'Robin Hood']
df_2.head()

df['is_duplicate_id'] = df.duplicated(['id'])

df_dupe_id_filter = df[df['is_duplicate_id'] == True]

df_dupe_id_filter.head()

df_3 = df[df['id'] == 42194]
df_3.head()

df.drop_duplicates(subset=['id'],inplace=True)

# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.

df.drop(['imdb_id', 'popularity', 'budget', 'revenue', 'homepage', 'tagline', 'overview', 'keywords', 'is_duplicate_title', 'is_duplicate_id'], axis=1, inplace=True)

df.head(1)

df['release_date'] = pd.to_datetime(df['release_date'])

# check it's worked
df.info()

df_genres = df.copy()
df_genres.drop(['cast', 'director', 'runtime', 'production_companies', 'budget_adj', 'revenue_adj'], axis=1, inplace=True)
df_genres.head(1)

# drop NaN values - only targets genres at this stage
df_genres.dropna(axis=0, how='any', inplace=True)

df_genres_split = df_genres['genres'].str[:].str.split('|', expand=True)

df_genres_split.head(10)

# 
df_genres_split.groupby(0).count()

genres_list = ['Action',
               'Adventure',
               'Animation',
               'Comedy', 
               'Crime', 
               'Documentary', 
               'Drama', 
               'Family', 
               'Fantasy', 
               'Foreign', 
               'History', 
               'Horror', 
               'Music', 
               'Mystery', 
               'Romance', 
               'Science Fiction', 
               'TV Movie', 
               'Thriller', 
               'War', 
               'Western']

count = 0
for i in genres_list:
    boolean_list = []
    genre_index = genres_list[count]
    for row in df_genres['genres']:
        if genre_index in row:
            boolean_list.append(True)
        else:
            boolean_list.append(False)
    df_genres[genres_list[count]] = boolean_list
    count = count + 1

df_genres.head(1)

df_genres.drop(['id', 'original_title', 'genres', 'release_date'], axis=1, inplace=True)

df_genres.head(1)

# df_genres_test = df_genres[df_genres.release_year != 2015]
# use this later for dropping specific 0 values from budget_adj

df_genres.describe()

df_genres = df_genres[df_genres['vote_count'] > 217.82]
df_genres = df_genres[df_genres['vote_average'] > 6.60]
df_genres

# create action dataframe
df_action = df_genres[df_genres['Action'] == True].copy()
df_action.drop(['Adventure', 
                'Animation', 
                'Comedy',
                'Crime',
                'Documentary', 
                'Drama', 
                'Family', 
                'Fantasy', 
                'Foreign', 
                'History', 
                'Horror', 
                'Music', 
                'Mystery', 
                'Romance', 
                'Science Fiction', 
                'TV Movie', 
                'Thriller', 
                'War', 
                'Western'], axis=1, inplace=True)
df_action.head(1)

# create adventure dataframe
df_adventure = df_genres[df_genres['Adventure'] == True].copy()
df_adventure.drop(['Action', 
                'Animation', 
                'Comedy', 
                'Crime',
                'Documentary', 
                'Drama', 
                'Family', 
                'Fantasy', 
                'Foreign', 
                'History', 
                'Horror', 
                'Music', 
                'Mystery', 
                'Romance', 
                'Science Fiction', 
                'TV Movie', 
                'Thriller', 
                'War', 
                'Western'], axis=1, inplace=True)
df_adventure.head(1)

# create animation dataframe
df_animation = df_genres[df_genres['Animation'] == True].copy()
df_animation.drop(['Action', 
                'Adventure', 
                'Comedy', 
                'Crime',
                'Documentary', 
                'Drama', 
                'Family', 
                'Fantasy', 
                'Foreign', 
                'History', 
                'Horror', 
                'Music', 
                'Mystery', 
                'Romance', 
                'Science Fiction', 
                'TV Movie', 
                'Thriller', 
                'War', 
                'Western'], axis=1, inplace=True)
df_animation.head(1)

# number of times a film is in the 'Crime' genre
df_crime = df_genres[df_genres['Crime'] == True]
crime_counts = df_crime['Crime'].count()
crime_counts



# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.



