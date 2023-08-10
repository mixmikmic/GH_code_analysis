get_ipython().run_line_magic('cat', 'some_file.txt')

fname = 'some_file.txt'

f = open(fname, 'r')
content = f.read()
f.close()

print(content)

fname = 'some_file.txt'
with open(fname, 'r') as f:
    content = f.read()

print(content)

fname = 'some_file.txt'
with open(fname, 'r') as f:
    content = f.readlines()

print(content)

fname = 'some_file.txt'
with open(fname, 'r') as f:
    for line in f:
        print(line)

fname = 'some_file.txt'
with open(fname, 'r') as f:
    for i, line in enumerate(f):
        print("Line {}: {}".format(i, line.strip()))

get_ipython().run_line_magic('cat', 'movie.json')

import json

fname = 'movie.json'
with open(fname, 'r') as f:
    content = f.read()
    movie = json.loads(content)

movie

import json

fname = 'movie.json'
with open(fname, 'r') as f:
    movie_alt = json.load(f)

movie == movie_alt

print(json.dumps(movie, indent=4))

get_ipython().run_line_magic('cat', 'movies-90s.jsonl')

import json

fname = 'movies-90s.jsonl'

with open(fname, 'r') as f:
    for line in f:
        try:
            movie = json.loads(line)
            print(movie['title'])
        except: 
            ...

get_ipython().run_line_magic('cat', 'data.csv')

import csv

fname = 'data.csv'

with open(fname, 'r') as f:
    data_reader = csv.reader(f, delimiter=',')
    headers = next(data_reader)
    print("Headers = {}".format(headers))
    for line in data_reader:
        print(line)

fname = 'data_no_header.csv'

with open(fname, 'r') as f:
    data_reader = csv.reader(f, delimiter=',')
    for line in data_reader:
        print(line)

fname = 'data.csv'

with open(fname, 'r') as f:
    data_reader = csv.reader(f, delimiter=',')
    headers = next(data_reader)
    data = []
    for line in data_reader:
        item = {headers[i]: value for i, value in enumerate(line)}
        data.append(item)

data

with open('movie.json', 'r') as f:
    content = f.read()
    data = json.loads(content)

data

type(data)

import pickle 

with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

get_ipython().run_line_magic('cat', 'data.pickle')

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

data

type(data)

import pandas as pd

get_ipython().run_line_magic('cat', 'movie.json')

data = pd.read_json('movie.json')
data.head()

get_ipython().run_line_magic('cat', 'movies-90s.jsonl')

data = pd.read_json('movies-90s.jsonl', lines=True)
data.head()

get_ipython().run_line_magic('cat', 'data.csv')

data = pd.read_csv('data.csv')
data.head()



