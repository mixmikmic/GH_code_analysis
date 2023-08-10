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

# since there are several hundred rows, create a new csv for easy viewing in excel
df_dupe_title_filter.to_csv('df_dupe_title_filter.csv', index=False)

# use this cell to spot check titles for differences
df_2 = df[df['original_title'] == 'Survivor']
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

# was hoping that this might help me get counts of all genre labels
# across all columns. Doesn't quite seem to work - if I change the 0 to 1, 2, 3 or 4
# then counts against the columns changes in a way that I don't quite know what is
# happening i.e. change to 1 and I hoped the value for Action in the 3 column would stay
# at 624... it doesn't it changes to 256
# the main benefit (by accident) is that I now know how many/what the different genre labels are
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

#df_genres.drop(['Comedey'], axis=1, inplace=True)
df_genres.head(1)

df_genres.drop(['genres'], axis=1, inplace=True)

df_genres.head(1)

# df_genres_test = df_genres[df_genres.release_year != 2015]
# use this later for dropping specific 0 values from budget_adj

# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.

# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.



