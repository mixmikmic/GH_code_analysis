
from nltk.twitter import Query, Streamer, Twitter, TweetViewer, TweetWriter, credsfromfile

oauth = credsfromfile()
client = Streamer(**oauth)
client.register(TweetViewer(limit=10))
client.sample()

client = Streamer(**oauth)
client.register(TweetViewer(limit=10))
client.filter(track='sad, happy')

client = Query(**oauth)
client.register(TweetViewer(limit=10))
client.user_tweets('GeorgeMonbiot', 10)

client = Query(**oauth)
tweets = client.search_tweets(keywords='nltk', limit=10)
tweet = next(tweets)
from pprint import pprint
pprint(tweet, depth=1)

client = Query(**oauth)
tweets = client.search_tweets(keywords='hillary clinton', limit=10)
tweet = next(tweets)

for tweet in tweets:
    print(tweet['text'])

client = Query(**oauth)
tweets = client.search_tweets(keywords='hillary clinton', limit=10000)
tweet = next(tweets)
lat = []
lon = []
myTweets = ""
for tweet in tweets:
    text = tweet['text']
    myTweets += ' ' + text
    coords = tweet["coordinates"]
    if (coords is not None):
        print(coords['coordinates'][0])
        print(coords['coordinates'][1])
        lat.append(coords['coordinates'][0])
        lon.append(coords['coordinates'][1])
        
        

get_ipython().magic('matplotlib inline')

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 14))
markersize = 400
markertype = '.'
markercolor = 'red'
markeralpha = 1

m = Basemap(projection='moll',lat_0=lat[0],lon_0=lon[0],resolution='l')
m.drawcoastlines(linewidth=0.25)

# Convert locations to x/y coordinates and plot them as dots.
x, y = m(lat, lon)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)
plt.show()

import nltk

def graph(myFile):
  tokens = nltk.tokenize.word_tokenize(myFile)
  fd = nltk.FreqDist(tokens)
  fd.plot(30,cumulative=False)

graph(myTweets)

from os import path
from wordcloud import WordCloud, STOPWORDS

# define stopwords: these will be excluded from the wordcloud
stopwords = set(STOPWORDS)
stopwords.add("https")

# Generate a word cloud image
wc = WordCloud(background_color="black", max_words=2000, stopwords=stopwords)
wc.generate(myTweets)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wc)
plt.axis("off")



