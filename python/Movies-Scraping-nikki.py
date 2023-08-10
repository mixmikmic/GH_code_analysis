# PART 1:  STANDALONE TO GRAB ALL MOVIES AND KEYWORDS

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
            results_p = requestResults("discover/movie?sort_by=popularity.desc&page=" + str(p))
            try:
                results_p = results_p[u'results']
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
                print('%d pages done' % p)
            except KeyError: 
                print('%d pages failed' % p)
                pass

### Run Part 1: REMEMBER TO CHANGE start_page to your start page, don't have to change increment
downloadMoviesToCSV(start_page=998, increment=200, filename='tmdb-movies-1001-to-1200.csv')

# PART 2: STANDALONE THAT TAKES IN .CSV FILE AND GETS ALL IMDB IDS in a separate file

import pandas as pd
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

def downloadIMDBIds(input_filename, output_filename):
    df = pd.read_csv(input_filename)

    with open(output_filename, 'w') as csvfile:
        fieldnames = ['id', 'imdb_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # TMDB limits 4 requests per second
        hit = 3 # Once hit reaches 0, call timer and reset hit to 3

        count = 0
        for tmid in df['id']:
            count += 1
            results = requestResults('movie/' + str(tmid) + '?x=1')
            if u'imdb_id' not in results or results[u'imdb_id'] is None:
                continue
            imid = results[u'imdb_id'].strip('tt')
            row = {'id': tmid, 'imdb_id': imid}
            writer.writerow(row)
            hit -= 1
            if hit <= 0:
                hit = 3
                time.sleep(1)
            if count % 200 == 0:
                print 'done with %d movies' % count

### Run Part 2: Get imdb ids from tmdb ids input csv file
downloadIMDBIds(input_filename='tmdb-movies-1001-to-1200.csv', output_filename='imdb-ids-1001-to-1200.csv')

# PART 3: STANDALONE THAT TAKES IN IMDB IDs and gets IMDB features
'''
Make sure you have IMDB installed.
- Go to: http://imdbpy.sourceforge.net/
- Download and unzip, then cd into it and make sure there is a setup.py file
- Run python setup.py install
- You're done! It's globally installed.
'''
import imdb
import pandas as pd
import csv
import requests
import numpy as np

def getIMDBFeatures(input_filename, output_filename, start, increment):

    # Note: This cannot be terminated via the stop button (interrupt the kernel), 
    # got to restart the kernel (use rewind button) :(
    
    ia = imdb.IMDb()
    df = pd.read_csv(input_filename)
    # Download increment movies at a time
    df = df[start:start+increment]
    
    imids = np.array(df['imdb_id'])

    with open(output_filename + '-' + str(start), 'w') as csvfile:
        # Grab these features from IMDB
        fieldnames = ['imdb_id', 'director', 'imdb_votes', 'certificate', 'num_stunts', 'num_fx']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for imid in imids:
            count += 1
            # Tries twice because sometimes it fails
            for i in range(2):
                try:
                    movie = ia.get_movie(str(int(imid)))
                    director = movie['director'][0]
                    imdb_votes = movie['votes']
                    certificate = movie['certificates'][-2].split(':')[1]
                    num_stunts = len(movie['stunt performer'])
                    num_fx = len(movie['special effects department'])
                    row = {'imdb_id': imid, 'director': director, 'imdb_votes': imdb_votes, 'certificate': certificate, 
                          'num_stunts': num_stunts, 'num_fx': num_fx}
                    writer.writerow(row)
                    break
                except:    
                    pass
            if count % 100 == 0:
                print 'Done with %d movies' % count
    print 'Done with page %d' % ((start%increment) + 1)

### Run Part 3: Get imdb features from imdb ids

# NOTE: This downloads 500 movies at a time and stores each in a different file.

import pandas as pd
df = pd.read_csv('imdb-ids-1001-to-1200.csv')
N = df.shape[0]
increment = 500 # Work on 500 movies at a time
end_page = N/increment

##############################################################
# NOTE: If you are done with page 2 (1000 movies), then change this to 2 the next time you start
start_page = 0
##############################################################

starts = [] 
for i in range(start_page,end_page): # default starts: [500,1000,1500,2000,2500,..,7500]
    starts.append((i+1)*increment)

for start in starts: 
    getIMDBFeatures(input_filename='imdb-ids-1001-to-1200.csv', output_filename='imdb-features-1001-to-1200.csv', 
                    start=start, increment=increment)

### PART 4: Merge all and output CSV file

import pandas as pd

# NOTE: Change to your filepath and start and end movie
prefix_filepath = 'leo-data/'
start = 1
end = 400

# Merge all imdb features into one
imdb_features = pd.read_csv(prefix_filepath + 'imdb-features-'+str(start)+'-to-'+str(end)+'.csv-500')
for p in range(2,16):
    imdb_features_ = pd.read_csv(prefix_filepath + 'imdb-features-'+str(start)+'-to-'+str(end)+'.csv-' + str(p*500))
    imdb_features = imdb_features.append(imdb_features_)

# Merge imdb ids with imdb features
imdb_ids = pd.read_csv(prefix_filepath + 'imdb-ids-'+str(start)+'-to-'+str(end)+'.csv')
imdb_ids = imdb_ids.rename(index=str, columns={"id": "tmdb_id"})
imdb_merged = imdb_ids.merge(imdb_features, how='outer', left_on='imdb_id', right_on='imdb_id')
imdb_merged = imdb_merged.dropna()

# Merge tmdb with imdb_merge
tmdb_movies = pd.read_csv(prefix_filepath + 'tmdb-movies-'+str(start)+'-to-'+str(end)+'.csv')
tmdb_movies = tmdb_movies.rename(index=str, columns={"id": "tmdb_id"})
full_movies = tmdb_movies.merge(imdb_merged, how='outer', left_on='tmdb_id', right_on='tmdb_id')
full_movies = full_movies.dropna()

# Output this to CSV of full movies
full_movies.to_csv('full-movies-'+str(start)+'-to-'+str(end)+'.csv', index=False)

### PART 5: Clean up columns to choose the right ones for Milestone 3
import pandas as pd
import numpy as np

df = pd.read_csv('full-movies-1-to-400.csv')

# Choose only columns we need
cols = ['genre_ids', 'poster_path', 'title', 'release_date', 'popularity', 'keywords', 'vote_count',
       'vote_average', 'director', 'imdb_votes', 'certificate', 'num_stunts', 'num_fx']
df = df[cols]

# Break down release date into month and year
datesplit = df['release_date'].str.split('-')
years = [int(d[0]) for d in datesplit]
months = [int(d[1]) for d in datesplit]
df['year'] = years
df['month'] = months
del df['release_date']

# TODO: Choose top 20 keywords / clustering, etc.
# TODO: Apply one hot encoding to multiple columns



