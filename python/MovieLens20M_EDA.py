import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df_gs = pd.read_csv('genome-scores.csv')

df_gs.shape

df_gs.head()

df_gs['relevance'].describe()

df_gs.groupby('movieId')['movieId'].count()

movies = pd.read_csv('movies.csv')

movies.tail()

movies.shape

ratings = pd.read_csv('ratings.csv')

ratings.shape

ratings.head(20)

ratings['rating'].describe()

sns.distplot(ratings['rating'])





new_df = df_gs.groupby('movieId')[['movieId']].count()

new_df.columns = ['dropmelater']

new_df.reset_index(inplace=True)

new_df.head()

new_df.shape

new_df.dtypes

ratings.dtypes

ratings_updated = ratings.join(new_df.set_index('movieId'), on='movieId', how='left', rsuffix='_drop_me_too')

ratings_updated.head()

ratings_updated.isnull().sum()

ratings_updated.dropna(axis=0, inplace=True)

ratings_updated.shape

ratings_updated.head()

ratings_updated.drop('dropmelater', axis=1, inplace=True)

ratings_updated.head()

ratings_updated.to_csv('ratings_updated.csv') #oops, forget to set_index to false and this takes a while

ratings_updated.groupby('userId')[['userId']].count()['userId'].describe()

movies_updated = movies.join(new_df.set_index('movieId'), on='movieId', how='left')

movies_updated.isnull().sum()

movies_updated.shape

movies_updated.dropna(axis=0, inplace=True)

movies_updated.drop('dropmelater', axis=1, inplace=True)

#  movies_updated['genres'].value_counts() 

### https://stackoverflow.com/questions/18889588/create-dummies-from-column-with-multiple-values-in-pandas
## This splits/dummies the strings in the genre column
dummies = pd.get_dummies(movies_updated['genres'])

atom_col = [c for c in dummies.columns if '|' not in c]

for col in atom_col:
    movies_updated[col] = dummies[[c for c in dummies.columns if col in c]].sum(axis=1)

movies_updated.columns

movies_updated.drop('genres',axis=1, inplace=True)

movies_updated.loc[movies_updated['(no genres listed)'] ==1]

# 'Adventure | Drama | Family | Mystery | Sci-Fi'

movies_updated.loc[24801, 'Adventure'] = 1
movies_updated.loc[24801, 'Drama'] = 1
movies_updated.loc[24801, 'Mystery'] = 1
movies_updated.loc[24801, 'Sci-Fi'] = 1

movies_updated.loc[24801]

movies_updated.drop('(no genres listed)', axis=1, inplace=True)

movies_updated.columns

movies_updated.to_csv('movies_updated.csv', index=False)

movies_updated.head()

users = ratings_updated.groupby('userId')[['userId']].count()

users.columns = ['rated_count']

ratings_updated.columns

users = users.join(ratings_updated.groupby('userId')[['rating']].mean())

users.sort_values('rated_count', inplace=True)

users.columns = ['rated_count', 'avg_rating']

users.reset_index(inplace=True)

users.shape

users.head()

users.to_csv('users.csv', index=False)

sns.distplot(users['avg_rating'], bins=5)



