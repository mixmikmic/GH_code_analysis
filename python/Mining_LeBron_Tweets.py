#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "your_access_token"
access_token_secret = "your_access_secret_token"
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'honda'
    stream.filter(track=['lebron','LeBron','LEBRON'])

get_ipython().run_line_magic('matplotlib', 'inline')
import json
import pandas as pd
import matplotlib.pyplot as plt
from os import path

d = path.dirname('/home/pybokeh/temp/')
#text = open(path.join(d, 'lebron_tweets_data.txt')).read()

tweets_data = []
tweets_file = open(path.join(d, 'lebron_tweets_data.txt'),'r')
for line in tweets_file:
    try:
        tweet = json.loads(line)
        if len(tweet) > 10:                 # Make sure we store complete tweets
            tweets_data.append(tweet)
    except:
        continue

print(len(tweets_data))

tweets_data[2]

tweets = pd.DataFrame()
tweets['text']      = [tweet['text'] for tweet in tweets_data]
tweets['lang']      = [tweet['lang'] for tweet in tweets_data]
tweets['retweeted'] = [tweet['retweeted'] for tweet in tweets_data]

tweets

english_tweets = tweets[(tweets['lang']=='en') & (tweets['retweeted']==False)]
english_tweets.drop_duplicates(subset='text');

text = ''
for line in english_tweets['text']:
    text = text + ' ' + line

get_ipython().run_line_magic('matplotlib', 'inline')
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
d = path.dirname('/home/pybokeh/Downloads/')
# Read the whole text.
#text = strWords
#text = open(path.join(d, 'alice.txt')).read()

"""
additional_words = [
'rt',
'ebay'
]

for word in additional_words:
    STOPWORDS.add(word)"""

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
# honda_mask = imread(path.join(d, "lebron_james_mask.png"), flatten=True)
# wc = WordCloud(background_color="black", max_words=2000, mask=honda_mask, stopwords=STOPWORDS)
wc = WordCloud(width=800, height=600).generate(text)

# generate word cloud
#wc.generate(text)

# store to file
wc.to_file(path.join(d, "lebron_word_cloud_simple.png"))

# show
plt.imshow(wc)
plt.axis("off")
#plt.figure()
#plt.imshow(honda_mask, cmap=plt.cm.gray)
#plt.axis("off")
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
d = path.dirname('/home/pybokeh/Downloads/')
# Read the whole text.
#text = strWords
#text = open(path.join(d, 'alice.txt')).read()

"""
additional_words = [
'rt',
'ebay'
]

for word in additional_words:
    STOPWORDS.add(word)"""

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
lebron_mask = imread(path.join(d, "lebron_james_mask.png"), flatten=True)
wc = WordCloud(background_color="black", max_words=2000, mask=lebron_mask, stopwords=STOPWORDS)

# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "lebron_word_cloud_masked.png"))

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(lebron_mask, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

wc.process_text(text)[:50]

STOPWORDS

