import pandas as pd

movies = pd.read_csv('./tmdb-movies.csv')

movies.head(1)

genre = movies.loc[:, ['id', 'genres']]
genre.head(5)

genre_list = genre['genres'].str.split('|').tolist()

for i in range(len(genre_list)):
    if not isinstance(genre_list[i], list):
        genre_list[i] = [genre_list[i]]

genre_list[:5]

stacked_genre = pd.DataFrame(genre_list, index=genre['id']).stack()

print(stacked_genre.head(1))

stacked_genre = stacked_genre.reset_index()

print(stacked_genre.head(1))

stacked_genre = stacked_genre.loc[:, ['id', 0]]

print(stacked_genre.head(5))

stacked_genre.columns = ['id', 'genre']

print(stacked_genre.head(1))

merged = pd.merge(movies, stacked_genre, on='id', how='left')

merged.head()

merged.info()

merged.sort_values(by=['id']).head()

merged.groupby('genre').count().sort_values(by=['id'], )



