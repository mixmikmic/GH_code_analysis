import pandas as pd

url = 'http://bit.ly/imdbratings'
movies = pd.read_csv(url)

# Looking at the first 5 rows of the DataFrame
movies.head()

# This will show descriptive statistics of numeric columns
movies.describe()

movies.describe(include=['float64'])

# Finding out dimensionality of DataFrame
movies.shape

# Finding out data types of each columns
movies.dtypes

type(movies)

