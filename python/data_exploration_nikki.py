# NOTE: Run this in Python 2.7
import requests
import urllib
import imdb
import lxml.html
import numpy as np
import pandas as pd
import itertools
get_ipython().magic('matplotlib inline')
import seaborn as sns
import math
import matplotlib.pyplot as plt

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
    id_ = imdb_access.search_movie(title)[0].__repr__().split('id:')[1].split('[')[0]
    hxs = lxml.html.document_fromstring(requests.get("http://www.imdb.com/title/tt" + id_).content)
    return hxs.xpath("//a[contains(@href, 'genre')]/text()")[1:]


# Get genres from IMDB for one movie by title
def imdbOverview(title):
    if (len(imdb_access.search_movie(title))>0):
        id_ = imdb_access.search_movie(title)[0].__repr__().split('id:')[1].split('[')[0]
        hxs = lxml.html.document_fromstring(requests.get("http://www.imdb.com/title/tt" + id_).content)
        text = hxs.xpath("//div[@class='summary_text']/text()")
        if (len(text)>0):
            return text[0].strip()
    return ''

# Get genres from TMDB for one movie by title
def tmdbGenresByTitle(title):
    title_query = urllib.urlencode({'query': title})
    genre_ids = requestResults("search/movie?" + title_query + "&language=en-US&page=1&include-adult=false")[u'results'][0]['genre_ids']
    return _mapGidsToGenres(genre_ids)

# Private helper function that maps genre_ids to genres
def _mapGidsToGenres(genre_ids):
    return [str(GENRE_MAP[gid]) for gid in genre_ids]

imdb_access = imdb.IMDb()

# EXAMPLES
# requestResults("discover/movie?sort_by=popularity.desc")[u'results'][0] # Get top 10 most popular results
# downloadImageToFile('/tnmL0g604PDRJwGJ5fsUSYKFo9.jpg', 't2s.jpg') # Download an image to the file t2s.jpg

imdbOverview('500 days of summer')

# Genre and movie poster path of your favorite movie from TMDB
fave_movie = 'Logan'

genres, poster_path = genreAndPosterPath(fave_movie)
print 'Fave movie: %s\n================\nGenres: %s\nPoster Path: %s' % (fave_movie, ', '.join(map(str, genres)), poster_path)

# Genre for this movie listed by TMDb and IMDb
imdb_genres_fave = imdbGenresByTitle(fave_movie)
tmdb_genres_fave = tmdbGenresByTitle(fave_movie)
print 'Genres of fave movie: %s\n================' % fave_movie
print 'IMDB: %s' % ', '.join(map(str, imdb_genres_fave))
print 'TMDB: %s' % ', '.join(map(str, tmdb_genres_fave))

# Challenge: Sci-Fi from IMDB is Science Fiction in TMDB! Need to find a genre mapping between both IMDB and TMDB.

# A list of the 10 most popular movies of 2016 from TMDb and their genre obtained via TMDB

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
#     return [{'title': str(r['title'].encode('ascii', 'ignore').decode('ascii')), 'genres': _mapGidsToGenres(r['genre_ids'])} for r in results]
#     return [{'title': str(r['title'].encode('ascii', 'ignore').decode('ascii')), 'keywords': get_keywords(int(r[u'id'])), 'genres': _mapGidsToGenres(r['genre_ids'])} for r in results]
    return results
def get_keywords(movieid):
    results = requestResults("movie/{}/keywords?x=1".format(movieid))
    keywords = []
    try:
        k = results[u'keywords']
        keywords = [str(r[u'name'].encode('ascii', 'ignore').decode('ascii')) for r in k]
    except KeyError:
        pass
    return keywords

get_keywords(321612)

# Print top 10 movies
top10movies_dict = top_n_movies_tmdb(10)
print 'Top 10 movies and their genres: \n================'
for m in top10movies_dict:
    print '%s: %s' % (m['title'], ', '.join(map(str, m['genres'])))

# genre pair matching between IMDB/TMBD for top 1000 movies 

# Pairs of genres that co occur for movies from TMDB

num_genres = len(GENRE_MAP.keys())
genre_matrix = np.zeros((num_genres, num_genres))

genres = list(map(str, GENRE_MAP.values()))
genre_df = pd.DataFrame(genre_matrix)
genre_df.columns = genres
genre_df.index = genres

# retrieve top 1000 movies
top_1000_movies = top_n_movies_tmdb(1000)
top_1000_movie_genres = map(lambda m: map(str,m['genres']), top_1000_movies)

# # create all pairs of genres
genre_pairs = list(itertools.combinations(genres, 2))
for genre in genres:
    genre_pairs.append(genre)
    
# for each pair find # movies with that pair:
for pair in genre_pairs:
    for movie_genres in top_1000_movie_genres:
        if set(list(pair)).issubset(movie_genres):
            genre_df[pair[0]][pair[1]] += 1
    

# plot heatmap of movie pairs that co-occur
plt.figure(figsize=(20,10))
sns.heatmap(genre_df, annot=True,fmt="g")

# genre bar chart counts for top 1000 movies
all_genres = list(itertools.chain.from_iterable(top_1000_movie_genres))
genres_unique, all_genres_counts = np.unique(all_genres, return_counts=True)
sorted_counts = [g for (g, c) in sorted(zip(all_genres_counts, genres_unique))]
sorted_genres = [c for (g, c) in sorted(zip(all_genres_counts, genres_unique))]

fig, ax = plt.subplots(figsize=(20, 10))
num_unique_genres = len(all_genres_counts)
ax.bar(range(num_unique_genres), sorted_counts)
ax.set_xticks(np.arange(num_unique_genres)+0.3)
ax.set_xticklabels(sorted_genres, rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# keywords by genre
genre_keywords = {}
for genre in genres:
    genre_keywords[genre] = {}

for movie in top_1000_movies:
    for genre in movie['genres']:
        for word in movie['keywords']:
            if not word in genre_keywords[genre]:
                genre_keywords[genre][word] = 1
            else:
                genre_keywords[genre][word] += 1
#         genre_keywords[genre] = genre_keywords[genre].union(movie['keywords'])    

sorted_genre_counts = {}
for genre in genres:
    sorted_genre_counts[genre] = sorted([(value,key) for (key,value) in genre_keywords[genre].items()], reverse=True)

for genre in genres:
    print genre + ' : ' + str([l for l in sorted_genre_counts[genre][:5]])

# top 200 movies popularity, average vote count, vote_average

popularity_top200 = []
vote_average_top200 = []
vote_count_top200 = []

top200_movies = []
top200_movies = top_n_movies_tmdb(200)


for movie in top200_movies:
    popularity_top200.append(movie['popularity'])
    vote_average_top200.append(movie['vote_average'])
    vote_count_top200.append(movie['vote_count'])
    
# plot histograms of popularity, vote average and vote count for top 200 movies
fig, axs = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(popularity_top200, kde=False, rug=True, ax=axs[0])
axs[0].set_xlabel('Popularity for top 200 movies')
axs[0].set_ylabel('Number of movies')
sns.distplot(vote_average_top200, kde=False, rug=True, ax=axs[1])
axs[1].set_xlabel('Vote Average for top 200 movies')
axs[1].set_ylabel('Number of movies')
sns.distplot(vote_count_top200, kde=False, rug=True, ax=axs[2])
axs[2].set_xlabel('Vote count for top 200 movies')
axs[2].set_ylabel('Number of movies')

movies = pd.read_csv('./tmdb-movies_start_601_end_801.csv')

titles = movies['original_title']

overviews = []
for i, title in enumerate(titles):
    if (i%100 ==0):
        print(i)
    ov = imdbOverview(title)
    overviews.append(ov)

movies['overviews'] = overviews

movies.to_csv('./tmdb-movies_start_601_end_801_w_overview.csv')



