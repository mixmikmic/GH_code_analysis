# Import libraries
import pandas as pd
import numpy as np

r_cols = ['user_id','movie_id','rating']
ratings = pd.read_csv('u.data',sep='\t', encoding="ISO-8859-1",header=None,names=r_cols,usecols=range(0,3))

m_cols = ['movie_id','movie_name']
movies = pd.read_csv('u.item',sep='|', encoding="ISO-8859-1",header=None,names=m_cols,usecols=range(0,2))
movies.head()

ratings=pd.merge(movies,ratings)

ratings.head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')

# Getting the number of ratings & average rating for each movie
movieStats = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
movieStats.columns = [' '.join(col).strip() for col in movieStats.columns.values]
movieStats.reset_index(inplace=True)
movieStats.head()

movieStats = pd.merge(movieStats[['movie_id','rating size','rating mean']],ratings[['movie_name','movie_id']],on='movie_id')
movieStats.drop_duplicates(inplace=True)

# Getting the popular movies - Popular defined as movies that have more than 100 ratings. 100 = parameter that can be modified
popular_movies = movieStats[movieStats['rating size'] > 100]
popular_movies.sort_values('rating size',ascending=False).head()

plt.figure(figsize=(10,4))
movieStats['rating size'].hist(bins=70)

plt.figure(figsize=(10,4))
movieStats['rating mean'].hist(bins=70)

sns.jointplot(x='rating mean',y='rating size',data=movieStats,alpha=0.5)

movieRatings = ratings.pivot_table(index=['user_id'],columns=['movie_name'],values='rating')
movieRatings.head()

starwars_user_ratings = movieRatings['Star Wars (1977)']
starwars_similarMovies = movieRatings.corrwith(starwars_user_ratings)
starwars_similarMovies = pd.DataFrame(starwars_similarMovies,columns=['similarity'])
starwars_similarMovies.dropna(inplace=True)
starwars_similarMovies.reset_index(inplace=True)
starwars_similarMovies.sort_values('similarity',ascending=False,inplace=True)
starwars_similarMovies.head(10)

starwars_similarMovies = pd.merge(starwars_similarMovies[['movie_name','similarity']],
                                  popular_movies[['movie_name','rating mean']],on='movie_name')

starwars_similarMovies.head(10)

liarliar_user_ratings = movieRatings['Liar Liar (1997)']
liarliar_similarMovies = movieRatings.corrwith(liarliar_user_ratings)
liarliar_similarMovies = pd.DataFrame(liarliar_similarMovies,columns=['similarity'])
liarliar_similarMovies.dropna(inplace=True)
liarliar_similarMovies.reset_index(inplace=True)
liarliar_similarMovies.sort_values('similarity',ascending=False,inplace=True)
liarliar_similarMovies.head(10)

liarliar_similarMovies = pd.merge(liarliar_similarMovies[['movie_name','similarity']],
                                  popular_movies[['movie_name','rating mean']],on='movie_name')

liarliar_similarMovies.head(10)



