import pandas as pd

df = pd.DataFrame([['German', 1777, 1855],
                   ['Swiss', 1707, 1783],
                   ['French', 1736, 1813],
                   ['French', 1749, 1827]],
                  index=['Gauss', 'Euler', 'Lagrange', 'Laplace'],
                  columns=['Nationality', 'Born', 'Died'])
df.index.name = 'Mathematician'
df

df.to_csv('math.csv')  # export to a csv file

get_ipython().system('cat math.csv  # run shell command to examine file')

del df  # delete the DataFrame
df = pd.read_csv('math.csv')  # read it back in
df

df = pd.read_csv('math.csv', index_col=0)  # use columns 0 as index
df

df.to_csv('math.csv', index=False, header=False)

get_ipython().system('cat math.csv')

df = pd.read_csv('math.csv', names=['Nationality', 'Born', 'Died'])  # read and supply columns names
df.index = ['Gauss', 'Euler', 'Lagrange', 'Laplace']   # supply the index
df.index.name = 'Mathematicians'
df

df.to_csv('math.csv')  # save again for index and header
chunks = pd.read_csv('math.csv', chunksize=1)  # chunksize is in no. of lines
french_count = 0
for piece in chunks:
    french_count += (piece['Nationality'].iloc[0] == 'French')
print 'Found %d French mathematicians.' % french_count

import sys
df.to_csv(sys.stdout, sep=':')  # can use a separator other than comma

import sqlite3

query = """
CREATE TABLE math
(Mathematician VARCHAR(10),
 Nationality VARCHAR(10),
 Born INTEGER,
 Died INTEGER
);"""

con = sqlite3.connect(':memory:')  # connect to an in-memory database
con.execute(query)  # execute the query
con.commit()  # commit the change

con.execute("INSERT INTO math VALUES('Gauss', 'German', 1777, 1855)")  # insert values
con.commit()  # and commit

cursor = con.execute('SELECT * FROM math')
for row in cursor:
    print row

values = [('Euler', 'Swiss', 1707, 1783),
          ('Lagrange', 'French', 1736, 1813),
          ('Laplace', 'French', 1749, 1827)]
con.executemany("INSERT INTO math VALUES(?, ?, ?, ?)", values)
con.commit()

cursor = con.execute('SELECT * FROM math')
print "Fetching one row..."
print cursor.fetchone()
print "Fetching all remaining rows..."
print cursor.fetchall()

import pandas.io.sql as sql
sql.read_sql('SELECT * FROM math', con)

sql.read_sql('SELECT * FROM math WHERE Nationality="French"', con)

from urllib2 import urlopen, Request  # to create HTTP requests and open URLs
import base64  # for base64 encoding
import json  # for handling the JSON format

consumer_key = 'dNcn9ZjPJ6dSaXJMYnVgna7jg'  # our app's consumer key
consumer_secret = open('consumer_secret', 'r').read().strip()  # read secret (should not be made public) from file

bearer_token = '%s:%s' % (consumer_key, consumer_secret)
encoded_bearer_token = base64.b64encode(bearer_token.encode('ascii'))  # bearer token needs to be base64 encoded
request = Request('https://api.twitter.com/oauth2/token')
request.add_header('Content-Type',
                   'application/x-www-form-urlencoded;charset=UTF-8')
request.add_header('Authorization',
                   'Basic %s' % encoded_bearer_token.decode('utf-8'))
request_data = 'grant_type=client_credentials'.encode('ascii')
request.add_data(request_data)

response = urlopen(request)  # make the request
raw_data = response.read().decode('utf-8')  # read the raw results in JSON format
data = json.loads(raw_data)  # decode JSON into Python data structures
bearer_token = data['access_token']  # extract the token

url = 'https://api.twitter.com/1.1/search/tweets.json?q=data%20science'  # search for "data science"
request = Request(url)
request.add_header('Authorization', 'Bearer %s' % bearer_token)  # use the bearer token from Step 2
response = urlopen(request)  # make the request
raw_data = response.read().decode('utf-8')  # results in raw JSON
data = json.loads(raw_data)  # decode JSON into Python data structures

print data.keys()

import pprint  # import pretty print module
pprint.pprint(data['statuses'][0])  # print the first tweet, it is a Python dict

# Extract the text and created_at fields and convert in pandas DataFrame
tweets_df = pd.DataFrame(data['statuses'], columns=['created_at', 'text'])
tweets_df

url = 'https://api.twitter.com/1.1/search/tweets.json?q=%23datascience'  # search for the hashtag #datascience
request = Request(url)
request.add_header('Authorization', 'Bearer %s' % bearer_token)  # use the bearer token from Step 2
response = urlopen(request)  # make the request
raw_data = response.read().decode('utf-8')  # results in raw JSON
data = json.loads(raw_data)  # decode JSON into Python data structures
hashtag_tweets_df = pd.DataFrame(data['statuses'], columns=['created_at', 'text'])
hashtag_tweets_df

# clean up temporary files in the end
get_ipython().system('rm math.csv')

