from tweepy import (Stream, OAuthHandler) # OAuth is an open standard for access delegation, commonly used as a way for Internet users to grant websites or applications access to their information on other websites but without giving them the passwords.
from tweepy.streaming import StreamListener
import time # For time.sleep()

C_KEY = 'INSERT_YOUR_OWN_HERE' # Consumer key
C_SECRET = 'INSERT_YOUR_OWN_HERE'
A_TOKEN = 'INSERT_YOUR_OWN_HERE' # Access token
A_SECRET = 'INSERT_YOUR_OWN_HERE'

class Listener(StreamListener):
    def on_status(self, status): # on_data() would print a lot more detailed data. on_status() focuses on status updates.
        try:
            save_file = open('twitDB.txt', 'a', encoding='utf-8') # a = append
            save_file.write(str(time.time()) + ':: ' + status.text.replace('\n', ' '))
            save_file.write('\n')
            save_file.close()
            return True
        except BaseException as err: # BaseException is the base class for all built-in exceptions. Problems that could happen are connection issues.
            print('Failed on_status, ', str(err))
            time.sleep(5)
    
    def on_error(self, status):
        print(status)
        
auth  = OAuthHandler(C_KEY, C_SECRET) # Authorizing ourselves
auth.set_access_token(A_TOKEN, A_SECRET)
twitter_stream = Stream(auth, Listener())
twitter_stream.filter(track='car') # Filtering tweets. Possible params: locations, languages, follow (people). The default argument for all of these is None. NB very few accounts have geolocations.

from tweepy import (Stream, OAuthHandler)
from tweepy.streaming import StreamListener
 
class Listener(StreamListener):

    tweet_counter = 0 # Static variable
    
    def login(self):
        CONSUMER_KEY = 'INSERT_YOUR_OWN_HERE'
        CONSUMER_SECRET = 'INSERT_YOUR_OWN_HERE'
        ACCESS_TOKEN = 'INSERT_YOUR_OWN_HERE'
        ACCESS_TOKEN_SECRET = 'INSERT_YOUR_OWN_HERE'

        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth
    
    def on_status(self, status):
        Listener.tweet_counter += 1
        print(str(Listener.tweet_counter) + '. Screen name = "%s" Tweet = "%s"'
              %(status.author.screen_name, status.text.replace('\n', ' ')))

        if Listener.tweet_counter < Listener.stop_at:
            return True
        else:
            print('Max num reached = ' + str(Listener.tweet_counter))
            return False

    def getTweetsByGPS(self, stop_at_number, latitude_start, longitude_start, latitude_finish, longitude_finish):
        try:
            Listener.stop_at = stop_at_number # Create static variable
            auth = self.login()
            streaming_api = Stream(auth, Listener(), timeout=60) # Socket timeout value
            streaming_api.filter(follow=None, locations=[latitude_start, longitude_start, latitude_finish, longitude_finish])
        except KeyboardInterrupt:
            print('Got keyboard interrupt')

    def getTweetsByHashtag(self, stop_at_number, hashtag):
        try:
            Listener.stopAt = stop_at_number
            auth = self.login()
            streaming_api = Stream(auth, Listener(), timeout=60)
            # Atlanta area.
            streaming_api.filter(track=[hashtag])
        except KeyboardInterrupt:
            print('Got keyboard interrupt')

listener = Listener()
listener.getTweetsByGPS(20, -84.395198, 33.746876, -84.385585, 33.841601) # Atlanta area. Tool to find coordinates for any place: http://boundingbox.klokantech.com/ (use CSV as the output format)

import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

punctuation += '´΄’…“”–—―»«' # string.punctuation misses these.

cache_english_stopwords = stopwords.words('english') # Could speed up code by making this a set

def tweet_clean(tweet):
    print('Original tweet:', tweet, '\n')
    # Remove HTML special entities (e.g. &amp;)
    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    print('No special entitites:', tweet_no_special_entities, '\n')
    # Remove tickers (Clickable stock market symbols that work like hashtags and start with dollar signs instead)
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities) # Substitute. $ needs to be escaped because it means something in regex. \w means alphanumeric char or underscore.
    print('No tickers:', tweet_no_tickers, '\n')
    # Remove hyperlinks
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    print('No hyperlinks:', tweet_no_hyperlinks, '\n')
    # Remove hashtags
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    print('No hashtags:', tweet_no_hashtags, '\n')
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_hashtags)
    print('No punctuation:', tweet_no_punctuation, '\n')
    # Remove words with 2 or fewer letters (Also takes care of RT)
    tweet_no_small_words = re.sub(r'\b\w{1,2}\b', '', tweet_no_punctuation) # \b represents a word boundary
    print('No small words:', tweet_no_small_words, '\n')
    # Remove whitespace (including new line characters)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words)
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ') # Remove single space left on the left
    print('No whitespace:', tweet_no_whitespace, '\n')
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if c <= '\uFFFF') # Apart from emojis (plane 1), this also removes historic scripts and mathematical alphanumerics (also plane 1), ideographs (plane 2) and more.
    print('No emojis:', tweet_no_emojis, '\n')
    # Tokenize: Change to lowercase, reduce length and remove handles
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) # reduce_len changes, for example, waaaaaayyyy to waaayyy.
    tw_list = tknzr.tokenize(tweet_no_emojis)
    print('Tweet tokenize:', tw_list, '\n')
    # Remove stopwords
    list_no_stopwords = [i for i in tw_list if i not in cache_english_stopwords]
    print('No stop words:', list_no_stopwords, '\n')
    # Final filtered tweet
    tweet_filtered =' '.join(list_no_stopwords) # ''.join() would join without spaces between words.
    print('Final tweet:', tweet_filtered)

s = '    RT @Amila #Test\nTom\'s newly listed Co. &amp; Mary\'s unlisted     Group to supply tech for nlTK.\nh.. $TSLA $AAPL https:// t.co/x34afsfQsh'
tweet_clean(s)

