# Import the libraries we will be using
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 17,12

np.random.seed(36)


### Read in the movies data
#movies = pd.read_csv("data/movies.dat", names=['movie_id', 'movie_title', 'genre'], 
#                      encoding='utf-8', sep="\:\:", engine='python')

### Movie ids don't start at 0 and some are missing, let's remap
#movie_id_map = dict(zip(np.argsort(movies['movie_id'].unique())*-1, movies['movie_id'].unique()))

### Given the mapping, let's replace the values
#movies = movies.replace({"movie_id": {v: k for k, v in movie_id_map.items()}})
#movies['movie_id'] = movies['movie_id'] * -1

#movies.tail()


### Read in the ratings data
#ratings = pd.read_csv("data/ratings.dat", names=['user_id', 'movie_id', 'rating', 'rating_timestamp'], 
#                      sep="\:\:", engine='python')

### User ids start at 1, let's bump them all down by 1
#ratings['user_id'] = ratings['user_id'] - 1

### Make movie ids match the ones from our movie's data
#ratings = ratings.replace({"movie_id": {v: k for k, v in movie_id_map.items()}})
#ratings['movie_id'] = ratings['movie_id'] * -1

### Put our mapping back in order
#movie_id_map = dict((key*-1, value) for (key, value) in movie_id_map.items())

#ratings.head()


#movies_ratings = pd.merge(movies, ratings, on="movie_id").drop(['genre', 'rating_timestamp'], axis=1)


#movies_ratings.tail()


#movies_ratings [ movies_ratings.movie_id == 22725 ]


#data = sparse.csr_matrix((movies_ratings['rating'], (movies_ratings['movie_id'], movies_ratings['user_id'])), 
#                         shape=(max(movies_ratings['movie_id'])+1, max(movies_ratings['user_id'])+1))

#### Format: rating in pairs of (movie, user) 

#data


#with open('data/movies_ratings.pickle', 'wb') as f:
#    pickle.dump(data, f)
#    f.close()

#with open('data/movies_clean.pickle', 'wb') as f:
#    pickle.dump(movies, f)
#    f.close()
    


with open('data/movies_ratings.pickle', 'rb') as f:
    data = pickle.load(f,encoding='latin1')
    f.close()

with open('data/movies_clean.pickle', 'rb') as f:
    movies = pickle.load(f,encoding='latin1')
    f.close()
    
print ("DATA REPRESENTATION: ")
print ("\n (Movie,User) Rating \n")
print (data[0:5])
print ("\n Movies information \n")
print (movies.head())


# D dimensional space

D = 2

svd = TruncatedSVD(D)
svd.fit(data)

svd


components = svd.transform(data)
components


plt.scatter(components[:,0], components[:,1])
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


def movie_search(movie_name_input):
    condition = movies.movie_title.str.contains(movie_name_input) 
    return movies[ condition ]


movie_search("The Matrix")


movie_search("Furious")


def movie_plotter(movie_id, components, x_buffer=3, y_buffer=3):
    
    # movie_id is the index, we want the 2 components
    
    x = components[movie_id][0]
    y = components[movie_id][1]
    
    # And we want all of the other movies with close values (range: less and greater)

    xs = [x - x_buffer, x + x_buffer]
    ys = [y - y_buffer, y + y_buffer]

    # Let's plot all the points and then only look at the zoom in that range (xs, ys)
    plt.scatter(components[:,0], components[:,1])
    plt.title('MOVIES WITH CLOSE COMPONENTS TO: '+ movies['movie_title'].loc[movie_id] +"\n", fontsize=14)
    
    plt.xlim(xs)
    plt.ylim(ys)

    # Include titles of movies in that range
    
    import re
    for x, y, title in zip(components[:,0], components[:,1], movies['movie_title']):
        if x >= xs[0] and x <= xs[1] and y >= ys[0] and y <= ys[1]:
            title_without_symbols = re.sub(r'[^\w]', ' ', title)
            plt.text(x, y, title_without_symbols)
            


id_to_plot_similar = 7613

movie_plotter( id_to_plot_similar, components )


id_to_plot_similar = 8574

movie_plotter( id_to_plot_similar, components, 1,1 )

