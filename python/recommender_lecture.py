import pandas as pd
import numpy as np

movies = pd.read_csv('../data/movies.csv')
ratings = pd.read_csv('../data/ratings.csv')
ratings.drop(['timestamp'], axis=1, inplace=True)

movies.head()

ratings.head()

def replace_name(x):
    return movies[movies['movieId']==x].title.values[0]

ratings.movieId = ratings.movieId.map(replace_name)

ratings.head()

M = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating')

M.shape

M

def pearson(s1, s2):
    """Take two pd.Series objects and return a pearson correlation."""
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

pearson(M['\'burbs, The (1989)'], M['10 Things I Hate About You (1999)'])

pearson(M['Harry Potter and the Sorcerer\'s Stone (a.k.a. Harry Potter and the Philosopher\'s Stone) (2001)'], 
        M['Harry Potter and the Half-Blood Prince (2009)'])

pearson(M['Mission: Impossible II (2000)'], M['Erin Brockovich (2000)'])

pearson(M['Clerks (1994)'],M['Mallrats (1995)'] )

def get_recs(movie_name, M, num):

    import numpy as np
    
    reviews = []
    for title in M.columns:
        if title == movie_name:
            continue
        cor = pearson(M[movie_name], M[title])
        if np.isnan(cor):
            continue
        else:
            reviews.append((title, cor))
    
    reviews.sort(key=lambda tup: tup[1], reverse=True)
    return reviews[:num]

    

recs = get_recs('Clerks (1994)', M, 10)

recs[:10]

anti_recs = get_recs('Clerks (1994)', M, 8551)

anti_recs[-10:]



