# NOTE: Run this in Python 2.7
import requests
import urllib
import imdb
import lxml.html

def requestResults(url):
    r = requests.get(BASE_URL + url + "&api_key=" + API_KEY)
    return r.json()

# Constants
BASE_URL = "https://api.themoviedb.org/3/"
API_KEY = "9767d17413ec9d9729c2cca238df02da"
GENRE_MAP = {}
for g in requestResults("genre/movie/list?x=1")[u'genres']:
    GENRE_MAP[g['id']] = g['name']

# Download image
def downloadImageToFile(imgpath, filename):
    # Does not return anything
    urllib.urlretrieve("https://image.tmdb.org/t/p/w500" + imgpath, filename)
    
# Get genre and poster path of one movie by title
def genreAndPosterPath(title):
    title_query = urllib.urlencode({'query': title})
    result = requestResults("search/movie?" + title_query + "&language=en-US&page=1&include-adult=false")[u'results'][0]
    genre_ids = result['genre_ids']
    genres = [str(GENRE_MAP[gid]) for gid in genre_ids]
    poster_path = result['poster_path']
    return genres, poster_path

# Get genres from IMDB for one movie by title
def imdbGenresByTitle(title):
    ###add
    imdb_access = imdb.IMDb()
    ##
    id_ = imdb_access.search_movie(title)[0].__repr__().split('id:')[1].split('[')[0]
    hxs = lxml.html.document_fromstring(requests.get("http://www.imdb.com/title/tt" + id_).content)
    return hxs.xpath("//a[contains(@href, 'genre')]/text()")[1:]

# Get genres from TMDB for one movie by title
def tmdbGenresByTitle(title):
    title_query = urllib.urlencode({'query': title})
    genre_ids = requestResults("search/movie?" + title_query + "&language=en-US&page=1&include-adult=false")[u'results'][0]['genre_ids']
    return _mapGidsToGenres(genre_ids)

# Private helper function that maps genre_ids to genres
def _mapGidsToGenres(genre_ids):
    return [str(GENRE_MAP[gid]) for gid in genre_ids]

# EXAMPLES
# requestResults("discover/movie?sort_by=popularity.desc")[u'results'][0] # Get top 10 most popular results
# downloadImageToFile('/tnmL0g604PDRJwGJ5fsUSYKFo9.jpg', 't2s.jpg') # Download an image to the file t2s.jpg

# Genre and movie poster path of your favorite movie from TMDB
fave_movie = 'Logan'

genres, poster_path = genreAndPosterPath(fave_movie)
print 'Fave movie: %s\n================\nGenres: %s\nPoster Path: %s' % (fave_movie, ', '.join(map(str, genres)), poster_path)

genres

poster_path

fave_movie2 = "Beauty and the beast"
genreAndPosterPath(fave_movie2)



# Genre for this movie listed by TMDb and IMDb
imdb_genres_fave = imdbGenresByTitle(fave_movie)
tmdb_genres_fave = tmdbGenresByTitle(fave_movie)
print 'Genres of fave movie: %s\n================' % fave_movie
print 'IMDB: %s' % ', '.join(map(str, imdb_genres_fave))
print 'TMDB: %s' % ', '.join(map(str, tmdb_genres_fave))

# Challenge: Sci-Fi from IMDB is Science Fiction in TMDB! Need to find a genre mapping between both IMDB and TMDB.



id = imdb_access.search_movie('Logan')[0].__repr__().split('id:')[1].split('[')[0]
id

hxs = lxml.html.document_fromstring(requests.get("http://www.imdb.com/title/tt" + id).content)
hxs

imdbGenresByTitle(fave_movie)



tmdbGenresByTitle(fave_movie)



def top10movies_tmdb():
    results = requestResults("discover/movie?sort_by=popularity.desc")[u'results'][:10]
    return [{'title': str(r['title']), 'genres': _mapGidsToGenres(r['genre_ids'])} for r in results]

# Print top 10 movies
top10movies_dict = top10movies_tmdb()
print 'Top 10 movies and their genres: \n================'
for m in top10movies_dict:
    print '%s: %s' % (m['title'], ', '.join(map(str, m['genres'])))



# Top 10 movies
top10movies = requestResults("discover/movie?sort_by=popularity.desc")[u'results'][1]
print(top10movies)
str(top10movies['original_language'])

str(requestResults("discover/movie?sort_by=popularity.desc")[u'results'][0]['poster_path'])

GENRE_MAP

imdb_access.search_movie("Logan")

imdb_access.search_movie("Logan")[0].keys()

imdb_access.search_movie("Logan")[0]['year']



get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

requestResults("discover/movie?sort_by=popularity.desc")[u'results'][0]

#three numeric columns 'popularity' + 'vote_average' + 'vote_count'
requestResults("discover/movie?sort_by=popularity.desc")[u'results'][0]['popularity']
#type is float

#Top10movies based on popularity
top200_movies = requestResults("discover/movie?sort_by=popularity.desc")[u'results'][:5]
type(top200_movies)

import math
def top_n_movies_tmdb(N):
    # only 20 results per page so need to send multiple request and increment page number 
    results_per_page = 20.0
    results = []
    num_pages = int(math.ceil(N/results_per_page))
    if num_pages == 1:
        results = requestResults("discover/movie?sort_by=popularity.desc")[u'results'][:N]
    else:
        for n in range(1, num_pages+1):
            result = requestResults("discover/movie?sort_by=popularity.desc&page={}".format(n))
            try:
                r= result[u'results']
                results = results + r
            except KeyError:
                pass
    return results

top200_movies = []
top200_movies = top_n_movies_tmdb(200)

population_top200 = []
vote_average_top200 = []
vote_count_top200 = []

for movie in top200_movies:
    population_top200.append(movie['popularity'])
    vote_average_top200.append(movie['vote_average'])
    vote_count_top200.append(movie['vote_count'])

#fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)
fig, axs = plt.subplots(1,3)
fig.set_size_inches(12, 5)
sns.distplot(population_top200, kde=False, rug=True, ax=axs[0])
sns.distplot(vote_average_top200, kde=False, rug=True, ax=axs[1])
sns.distplot(vote_count_top200, kde=False, rug=True, ax=axs[2])

m,b = np.polyfit(vote_average_top200, population_top200, 1)
print(m)
print(b)

x = vote_average_top200
y = population_top200

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')

top200_movies[0]

type(top200_movies[0]['adult'])

top200_movies[0]['genre_ids']

#adults % by genre
adult = []
genre_adult = dict()
for movie in top200_movies:
    adult_bool = movie['adult']
    adult.append(adult_bool)
    if adult_bool:
        genre_ids = movie['genre_ids']
        for genre_id in genre_ids:
            if genre_id in genre_adult:
                genre_adult[genre_id] += 1
            else:
                genre_adult[genre_id] = 1

sum(adult) #NO ADULT

#language % by genre
str(top200_movies[0]['original_language'])

languages = []
genre_language = dict()
for movie in top200_movies:
    language = str(movie['original_language'])
    if language not in languages:
        languages.append(language)
    else:
        pass
    #genre_ids = movie['genre_ids']

languages

def language_genre(lan):
    lan_genre = dict()
    for movie in top200_movies:
        language = str(movie['original_language'])
        genre_ids = movie['genre_ids']
        if language == lan:
            for genre_id in genre_ids:
                if genre_id in genre_adult:
                    lan_genre[genre_id] += 1
                else:
                    lan_genre[genre_id] = 1     
    return lan_genre

for lan in languages:
    print(language_genre(lan))









