import os
import json

DATA_DIR = os.path.join('data', 'boxofficemojo')
files=os.listdir(DATA_DIR)

movies = []
for file in files :
    file_path= os.path.join(DATA_DIR,file)
    with open(file_path,'r') as target_files:
        movie = json.load(target_files)
        #print movie
    movies.append(movie)

movies.head()

print len(movies)

import pandas as pd
movies_df = pd.DataFrame(movies)

movies_df.head()

movies_df['year'].hist()

movies_df.plot.hist('year')

DATA_DIR2 = os.path.join('data', 'metacritic')
cric_files=os.listdir(DATA_DIR2)

cric_files[:10]

crics = []
for file in cric_files :
    file_path= os.path.join(DATA_DIR2,file)
    with open(file_path,'r') as target_files:
        cric = json.load(target_files)
        #print movie
    crics.append(cric)

print len(crics)

crics[:10]


crics_df = pd.DataFrame(crics)

movies[:10]

crics[0]

type(crics)

type(crics[0])

movies_df['year'].count()

movies_df.describe()

movies_df.info()

movies_df['production_budget'].nlargest(10)

N=4
top_directors = movies_df.director.value_counts().index[:N]
top_dir_movies = movies_df[movies_df['director'].isin(top_directors)]

top_dir_movies.head()



