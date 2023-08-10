get_ipython().system('head -n5 movies.csv')

get_ipython().system('head -n5 ratings_5M.csv')

get_ipython().system('wc -l ml-latest-small/movies.csv')
get_ipython().system('wc -l ml-latest-small/ratings.csv')

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ratings_500K.csv', 
                      names=r_cols, 
                      usecols=range(3), 
                      header=None, 
                      low_memory=False, 
                      dtype={'user_id':'int', 
                             'movie_id':'int',
                             'rating':'float'})
ratings.head()

m_cols = ['movie_id', 'title']
movies = pd.read_csv('movies.csv', 
                     names=m_cols, 
                     usecols=range(2), 
                     header=None, 
                     low_memory=False, 
                     dtype={'movie_id':'int',
                            'user_id':'int'})

movies.head()

ratings = pd.merge(movies, ratings)

ratings.head()

ratings.info()

movies.info()

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

corrMatrix = userRatings.corr()
corrMatrix.head()

corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()

myRatings = userRatings.loc[1].dropna()
myRatings

similar_candidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print("Adding similar movies for " + myRatings.index[i] + "...")
    # retrieve similar movies to this one that I rated
    similar_movies = corrMatrix[myRatings.index[i]].dropna()
    # scale its similarity by how well I rated this movie
    similar_movies = similar_movies.map(lambda x: x * myRatings[i])
    # add the score to the list of similar candidates
    similar_candidates = similar_candidates.append(similar_movies)
    
print("Sort recommendations...")
similar_candidates.sort_values(inplace = True, ascending = False)
similar_candidates.head(10)

similar_candidates = similar_candidates.groupby(similar_candidates.index).sum()

similar_candidates.sort_values(inplace = True, ascending = False)
similar_candidates.head(10)

