import pandas as pd
from collections import Counter
import re
import stopwords as st

get_ipython().magic('run tweetering.py trump 250     #save about 250 tweets containing "trump" or "#trump"')

get_ipython().magic('run tweetering.py clinton 250       #save about 250 tweets containing "clinton" or "#clinton"')

def processTweet(tweet):
    
    if tweet.startswith("RT"):
        i = tweet.index(':')
        tweet = tweet[i+2:]
    
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('([0-9]+)','', tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('&amp;', '', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

stoplist = ["clinton","trump","hillary","donald",'chair',"rt",":","oh","clinton.","trump,","team",".","bill","like","made","now","trump:","via","\xe2\x80\x94","de","\xe2\x80\x93","republican","will","going","-","campaign","election","us","things]","president",
            "camp","seen","--","well","breaking:","tv","november","media","anonymous","video","ng","can","just","girl","calls","ready","trump\'s","clinton\'s","saying","really","girl","calls","former","electing","teen",
           "say","presidential","%","obama","new","take","talk","vote","people","may","watch","test","voters:","must","gop","live","breaks",
           "poll","yay","one","trump’s","voting","nothing","trump.","real","old","back"]

def countinfile(filename):
    d = {}
    stopwords = st.get_stopwords("en") + stoplist
    with open(filename, "r") as fp:
        for line in fp:
            line = processTweet(line)
            #print line
            words = line.strip().split()
            for word in words:
                try:
                    if(word not in stopwords):
                        d[word] += 1
                except KeyError:
                    d[word] = 1
    return d

dict_clinton = countinfile("clinton.txt")
dict_trump = countinfile("trump.txt")

data1 = Counter(dict_trump).most_common(10)
data2 = Counter(dict_clinton).most_common(10)

df_trump = pd.DataFrame(data1, columns=["Word","Frequency"])
df_clinton = pd.DataFrame(data2, columns=["Word","Frequency"])

print "Trump:"
print df_trump
print "\nClinton:"
print df_clinton

df_trump["Sentiment"] = [-1,-1,1,-1,1,1,-1,-1,-1,1]
df_clinton["Sentiment"] = [-1,1,-1,-1,-1,-1,-1,1,-1,1]

print "Trump:"
print df_trump
print "\nClinton:"
print df_clinton

sum1 = 0
for i in range(len(df_trump)):
    sum1 += df_trump["Frequency"][i] * df_trump["Sentiment"][i]
print "Trump's average sentiment: " + str(float(sum1) / 10)

sum2 = 0
for i in range(len(df_clinton)):
    sum2 += df_clinton["Frequency"][i] * df_clinton["Sentiment"][i]
print "Clinton's average sentiment: " + str(float(sum2) / 10)

