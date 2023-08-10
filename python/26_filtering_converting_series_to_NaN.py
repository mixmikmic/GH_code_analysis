import pandas as pd

url = 'http://bit.ly/imdbratings'
movies = pd.read_csv(url)

movies.head()

# counting missing values
movies.content_rating.isnull().sum()

movies.loc[movies.content_rating.isnull(), :]

# counting content_rating unique values
# you can see there're 65 'NOT RATED' and 3 'NaN'
# we want to combine all to make 68 NaN
movies.content_rating.value_counts(dropna=False)

# examining content_rating's 'NOT RATED'
movies.loc[movies.content_rating=='NOT RATED', :]

# filtering only 1 column
movies.loc[movies.content_rating=='NOT RATED', 'content_rating']

import numpy as np

type(movies.loc[movies.content_rating=='NOT RATED', 'content_rating'])

# there's no error here
# however, if you use other methods of slicing, it would output an error

# equating this series to np.nan converts all to 'NaN'
movies.loc[movies.content_rating=='NOT RATED', 'content_rating'] = np.nan

# it has changed from 65 to 68
movies.content_rating.isnull().sum()

# select top_movies
top_movies = movies.loc[movies.star_rating >= 9, :]

top_movies

# there's a SettingWithCopyWarning here because Pandas is not sure if the DataFrame is a view or copy
top_movies.loc[0, 'duration'] = 150

top_movies

# to get rid of the error, always use .copy()

top_movies = movies.loc[movies.star_rating >= 9, :].copy()

top_movies.loc[0, 'duration'] = 150

