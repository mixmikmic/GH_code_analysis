import numpy as np
import pandas as pd

from pandas import Series, DataFrame

get_ipython().system('cat data/sample.csv')

pd.read_csv('data/sample.csv') # or pd.read_table('data/sample.csv', sep=',')

pd.read_csv('data/data_with_comments.csv',skiprows=[0,1])

pd.read_csv('data/sample.csv', na_values=[5])

# reading large files in chunks
chunker = pd.read_csv('data/sample.csv', chunksize=1)
total = Series([])
for piece in chunker:
    total = total.add(piece['male'].value_counts(), fill_value=0)
    
total.sort_values(ascending=False)

# search python pandas in Twitter
import requests
url = 'http://search.twitter.com/search.json?q=python%20pandas'
resp = requests.get(url)

# then parse the http response
import json
data = json.loads(resp.text)

# create a data frame from the tweets
tweet_fields = ['created_at', 'from_user', 'id', 'text']
tweets = DataFrame(data['results'], columns=tweet_fields)





