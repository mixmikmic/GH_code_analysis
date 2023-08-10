# Import the libraries we will be using
import pickle
from sklearn.decomposition import TruncatedSVD
from moakler.movies import movie_search
from moakler.movies import movie_plotter

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 14, 8

with open('data/movies_ratings.pickle', 'rb') as f:
    data = pickle.load(f)

with open('data/movies_clean.pickle', 'rb') as f:
    movies = pickle.load(f)

svd = TruncatedSVD(2)

svd.fit(data)

components = svd.transform(data)

movie_plotter(components, movies)

movie_search(movies, "Jurassic")

movie_plotter(components, movies, 10163, 100, 100)





