import pandas as pd

url = 'http://bit.ly/imdbratings'

# Create movies DataFrame
movies = pd.read_csv(url)

movies.head()

movies[movies.duration >= 200]

True or False

True or True

False or False

True and True

True and False

# when you wrap conditions in parantheses, you give order
# you do those in brackets first before 'and'
# AND
movies[(movies.duration >= 200) & (movies.genre == 'Drama')]

# OR 
movies[(movies.duration >= 200) | (movies.genre == 'Drama')]

(movies.duration >= 200) | (movies.genre == 'Drama')

(movies.duration >= 200) & (movies.genre == 'Drama')

# slow method
movies[(movies.genre == 'Crime') | (movies.genre == 'Drama') | (movies.genre == 'Action')]

# fast method
filter_list = ['Crime', 'Drama', 'Action']
movies[movies.genre.isin(filter_list)]

