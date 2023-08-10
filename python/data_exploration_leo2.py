# NOTE: Run this in Python 2.7
import requests
import urllib
import lxml
import imdb
from IPython.display import Image
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
    
def movieDetails(title):
    title_query = urllib.urlencode({'query': title})
    result = requestResults("search/movie?" + title_query + "&language=en-US&page=1&include-adult=false")[u'results'][0]
    return result
    
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
    imdb_access = imdb.IMDb()
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

# Genre for this movie listed by TMDb and IMDb
imdb_genres_fave = imdbGenresByTitle(fave_movie)
tmdb_genres_fave = tmdbGenresByTitle(fave_movie)
print 'Genres of fave movie: %s\n================' % fave_movie
print 'IMDB: %s' % ', '.join(map(str, imdb_genres_fave))
print 'TMDB: %s' % ', '.join(map(str, tmdb_genres_fave))

# Challenge: Sci-Fi from IMDB is Science Fiction in TMDB! Need to find a genre mapping between both IMDB and TMDB.

# A list of the 10 most popular movies of 2016 from TMDb and their genre obtained via TMDB

def top10movies_tmdb():
    results = requestResults("discover/movie?sort_by=popularity.desc")[u'results'][:10]
    return [{'title': str(r['title']), 'genres': _mapGidsToGenres(r['genre_ids'])} for r in results]

# Print top 10 movies
top10movies_dict = top10movies_tmdb()
print 'Top 10 movies and their genres: \n================'
for m in top10movies_dict:
    print '%s: %s' % (m['title'], ', '.join(map(str, m['genres'])))

# Word cloud of keywords of top movies from year to year
def getKeywordsByYear(year):
    top_movies = requestResults("discover/movie?sort_by=popularity.desc&year=" + str(year))[u'results']
    top_ids = [m[u'id'] for m in top_movies]
    keywords = []
    for movie_id in top_ids:
        keywords += _getKeywordsById(movie_id)
    return keywords

# Private helper function that gets keywords by movie id
def _getKeywordsById(movie_id):
    keywords_dict = requestResults("movie/" + str(movie_id) + "/keywords?language=en-US")[u'keywords']
    return [str(k[u'name']) for k in keywords_dict]

keywords_1995 = getKeywordsByYear(1995)
keywords_2000 = getKeywordsByYear(2000)
keywords_2005 = getKeywordsByYear(2005)
keywords_2010 = getKeywordsByYear(2010)
keywords_2015 = getKeywordsByYear(2015)

# Using https://www.jasondavies.com/wordcloud/, we plotted word clouds of the keywords of top movies in each year

# Keywords in 1995
Image("keywords_1995.png")

# Keywords in 2000
Image("keywords_2000.png")

# Keywords in 2005
Image("keywords_2005.png")

# Keywords in 2010
Image("keywords_2010.png")

# Keywords in 2015
Image("keywords_2015.png")

genreCountsByYears = [] # Initialize it here first

def genreCountByYear(year):
    genre_count = {}
    for p in range(1,26): # Top 500 movies
        results_p = requestResults("discover/movie?sort_by=popularity.desc&page=" + str(p) + "&year=" + str(year))[u'results']
        for m in results_p:
            for g in m[u'genre_ids']:
                # Map g to genre name
                if g not in GENRE_MAP:
                    continue
                g_name = str(GENRE_MAP[g])
                if g_name not in genre_count:
                    genre_count[g_name] = 1
                else:
                    genre_count[g_name] = genre_count[g_name] + 1
    return genre_count

# Compute genre counts of each year from 1995 to 2016
# Run them one by one as they are async

genreCountsByYears.append(genreCountByYear(2016))

# Compare horror and comedy across the years
horror_counts = [gy['Horror'] for gy in genreCountsByYears]
comedy_counts = [gy['Comedy'] for gy in genreCountsByYears]

plt.title('Horror vs Comedy')
plt.plot(horror_counts, label='Horror')
plt.plot(comedy_counts, label='Comedy')
plt.legend(loc='best')
plt.show()
plt.grid()

# Figure out top and bottom genres across the years

top_genres = [max(gy, key=lambda k: gy[k]) for gy in genreCountsByYears]
top_genres
print 'Top genre every year is: Drama'

bottom_genres = [min(gy, key=lambda k: gy[k]) for gy in genreCountsByYears]
bottom_genres
print 'Bottom genres toggle between: TV Movie, Western, Documentary'

# Plot the growth of a few genres
plt.figure(figsize=(25,12))
action_counts = [gy['Action'] for gy in genreCountsByYears]
adventure_counts = [gy['Adventure'] for gy in genreCountsByYears]
animation_counts = [gy['Animation'] for gy in genreCountsByYears]
romance_counts = [gy['Romance'] for gy in genreCountsByYears]
war_counts = [gy['War'] for gy in genreCountsByYears]
plt.title('Action, Adventure, Animation, Romance, War')
plt.plot(action_counts, label='Action')
plt.plot(adventure_counts, label='Adventure')
plt.plot(animation_counts, label='Animation')
plt.plot(romance_counts, label='Romance')
plt.plot(war_counts, label='War')
plt.legend(loc='best')
plt.show()

# STANDALONE TO GRAB ALL MOVIES AND KEYWORDS

import csv
import time
import requests


#########################################################
'''
BASE STUFF THAT IS ALSO DEFINED ON TOP
'''
def requestResults(url):
    r = requests.get(BASE_URL + url + "&api_key=" + API_KEY)
    return r.json()

# Constants
BASE_URL = "https://api.themoviedb.org/3/"
API_KEY = "9767d17413ec9d9729c2cca238df02da"
GENRE_MAP = {}
for g in requestResults("genre/movie/list?x=1")[u'genres']:
    GENRE_MAP[g['id']] = g['name']
    
#########################################################

def _getKeywordsStringById(movie_id):
    
    keywords_dict = requestResults("movie/" + str(movie_id) + "/keywords?language=en-US")
    if u'keywords' not in keywords_dict:
        return ''
    keywords_dict = keywords_dict[u'keywords']
    kstring = ''
    for k in keywords_dict:
        kstring += k[u'name'] + ','
    return str(kstring.encode('utf-8').strip())[:-1]

def _tidyRow(m, keywords):
    # Makes sure the row of movie is well-formatted
    output = {}
    for k in m:
        typem = type(m[k])
        k = str(k)
        if typem == str or typem == unicode:
            output[k] = m[k].encode('utf-8').strip()
        else:
            output[k] = m[k]
    output['keywords'] = keywords
    return output

def downloadMoviesToCSV(start_page, increment, filename):
    genre_count = {}
    
    with open(filename, 'w') as csvfile:
        fieldnames = ['id', 'genre_ids', 'poster_path', 'title', 'overview', 'release_date', 
                      'popularity', 'original_title', 'backdrop_path', 'keywords', 
                     'vote_count', 'video', 'adult', 'vote_average', 'original_language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Get keywords for movies
        writer.writeheader()
        
        # TMDB limits 4 requests per second
        hit = 3 # Once hit reaches 0, call timer and reset hit to 3
        
        for p in range(start_page,start_page+increment): 
            if p%10 == 0:
                print('page: ', p)
            results_p = requestResults("discover/movie?sort_by=popularity.desc&page=" + str(p))[u'results']
            hit -= 1
            if hit <= 0:
                hit = 3
                time.sleep(1)

            # Write to CSV
            
            for m in results_p:
                mid = m[u'id']
                keywords = _getKeywordsStringById(mid)
                hit -= 1
                if hit <= 0:
                    hit = 3
                    time.sleep(1)
                
                row = _tidyRow(m, keywords)
                writer.writerow(row)

#             time.sleep(0.5) # We hit 2 requests per second without the keywords

downloadMoviesToCSV(start_page=601, increment=200, filename='tmdb-movies_601.csv')



