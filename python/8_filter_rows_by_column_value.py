import pandas as pd

# url

url = 'http://bit.ly/imdbratings'

# create DataFrame called movies
movies = pd.read_csv(url)

movies.head()

movies.shape

# booleans
type(True)
type(False)

# create list
booleans = []

# loop
for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)

booleans[0:5]

# len(booleans) is the same as the number of rows in movies' DataFrame
len(booleans)

# convert booleans into a Pandas series
is_long = pd.Series(booleans)

is_long.head()

# pulls out genre
movies['genre']

# this pulls out duration >= 200mins
movies[is_long]

# this line of code replaces the for loop
# when you use a series name using pandas and use a comparison operator, it will loop through each row
is_long = movies.duration >= 200
is_long.head()

movies[is_long]

movies[movies.duration >= 200]

# this is a DataFrame, we use dot or bracket notation to get what we want
movies[movies.duration >= 200]['genre']
movies[movies.duration >= 200].genre

# best practice is to use .loc instead of what we did above by selecting columns
movies.loc[movies.duration >= 200, 'genre']

