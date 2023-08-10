#!/usr/bin/python

from IPython.display import display, HTML 
import tweepy
import csv
#import urllib3.contrib.pyopenssl
import urllib
get_ipython().magic('matplotlib inline')


#urllib3.contrib.pyopenssl.inject_into_urllib3()



#For utf-8 encoding of the tweet, getting it's sentiment through the api.
def tweet_polarity(twt):
    twt = twt.text.encode("utf-8")
    d1 = urllib.urlencode({"text": twt})
    p = urllib.urlopen("http://text-processing.com/api/sentiment/", d1)
    tp = p.read()
    return tp


#Twitter keys for authentication
consumer_key = ''
consumer_secret = ''
access_key = ''
access_secret = ''


#Setting up authentication
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)



# Open/Create a file to append data
csvFile = open('result_limited.csv', 'a')
csvWriter = csv.writer(csvFile)

#Score is the guage using which we calculate the trend of sentiment of the organization.
#For simplicity, score for (negitive sentiment = -1), (positive sentiment = +1) and (neutral sentiment fetches 0)

#Would be used for real-time plotting of sentiment.
score = 0

#Only for the sake of POC. Static line graph plot of sentiment.
arr = []

# Can set the since and until dates for the stream. Disabling/not including it would get us live stream.
#q => search term
#Limiting the items to 10. Not including it would give a flowing stream.  (Limited to 10 for demo purpose)
for tweet in tweepy.Cursor(api.search,
                    q="Kayako",
                    #since="2014-02-14",
                    #until="2014-02-15",
                    lang="en").items(10):

    #Writes a row to the csv file. Used for generating a csv file for analysis.
    #csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'), tweet_polarity(tweet)])
    #print tweet.created_at, tweet.text.encode('utf-8'), tweet_polarity(tweet)
    
    #print tweet_polarity(tweet)[-10:]
    
    print tweet.text.encode('utf-8'), tweet_polarity(tweet)
    if "pos" in tweet_polarity(tweet)[-10:]:
        score += 1
        arr.append(1)
    elif "neg" in tweet_polarity(tweet)[-10:]:
        score -= 1
        arr.append(-1)
    else:
        score += 0
        arr.append(0)
    
    print("  ")
    print("  ")
csvFile.close()

print score
print arr


#Things to remember:#
#                   #
#                   #
#Text should not exceed 80k characters.

#Languages supported:
#arabic, english, danish, dutch, finnish, french, german, hungarian, italian, norwegian, portuguese, romanian
#russian, spanish, swedish


import seaborn as sns
sns.set(palette="Set2")

ax = sns.tsplot(arr, err_style="ci_band", ci = 95)
ax.set_ylim(-2, 2)



