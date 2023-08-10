import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import nltk.classify.util
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import IPython
import re
from bs4 import BeautifulSoup

plt.style.use('fivethirtyeight') #538

#the %matplotlib inline will make our plot outputs appear and be stored within the notebook.
#%matplotlib.inline  

#For anyone stumbling across this, as of PR #3381, you can now enable 2x images by just adding the line:
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

# reading the dataset .csv file
cols = ['sentiment','id','date','query_string','user','text']
dataframe = pd.read_csv("training_data.csv", header=None ,names=cols )
dataframe.head()

dataframe.describe()

dataframe.info()

#by closely watching the dataset we get to know that id , data , query_string , user are of no use 
# axis = 1 mean column
dataframe= dataframe.drop(['id','date','query_string','user'],axis=1)
dataframe[dataframe.sentiment == 0].index

dataframe.head()

dataframe['pre_clean_len'] = [len(t) for t in dataframe.text]

dataframe.head()

max(dataframe.pre_clean_len)

# since the max no of character in the twitter can be 140 but here it is 374  that is the 
# the text is not clean we need to clean it 

#cleaning HTML tags using BS4
dataframe.text[279]
# output-- "Whinging. My client&amp;boss don't understand English well. 
# Rewrote some text unreadable. It's written by v. good writer&amp;reviewed correctly. "
ex = BeautifulSoup(dataframe.text[279],'lxml')
ex.get_text()
# output -- u"Whinging. My client&boss don't understand English well.
# Rewrote some text unreadable. It's written by v. good writer&reviewed correctly. "

#cleaning @mentions because they are not usefull
dataframe.text[343]
# output-- '@TheLeagueSF Not Fun &amp; Furious? The new mantra for the Bay 2 Breakers? 
# It was getting 2 rambunctious;the city overreacted &amp; clamped down '
re.sub(r'@[a-zA-Z0-9]+','',dataframe.text[343])
# output -- ' Not Fun &amp; Furious? The new mantra for the Bay 2 Breakers? 
# It was getting 2 rambunctious;the city overreacted &amp; clamped down '

#removing URL links
dataframe.text[0]
# output-- "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
re.sub(r'https?://[a-zA-Z0-9./]+','',dataframe.text[0])
# "@switchfoot  - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
# removing non ascii char
dataframe.text[226].decode('UTF-8')
re.sub(u'\ufffd',' ',dataframe.text[226].decode("UTF-8"))

# tough #_tag are important so we cant remove words withs #_tags but we can remove the pucn tuations #
dataframe.text[175]
re.sub(r'[^a-zA-za]',' ',dataframe.text[175])

#cleaning our dataset
combine_mention_and_URLs = r'@[A-za-z0-9]+|https?://[a-zA-Z0-9./]+'

#dealing with web links like www.
ww_remov = r'www.[^ ]+'

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text,'lxml')
    souped = soup.get_text()
    stripped = re.sub(combine_mention_and_URLs,'',souped)
    stripped = re.sub(ww_remov, '', stripped)
    
    try:
        clean = stripped.decode("utf-8").replace(u"fffd"," ")
    except:
        clean = stripped
    
    lowercase = clean.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lowercase)
    letters_only = re.sub(r"[^a-zA-Z]"," ",neg_handled)
    words = word_tokenize(letters_only)
    return (" ".join(words)).strip()

testing = dataframe.text[:10]
test_rsult = []
for t in testing:
    test_rsult.append(tweet_cleaner(t))
test_rsult[:10]

get_ipython().run_cell_magic('time', '', 'nums = [0,40000,800000,840000]\n\nprint "cleaning and parsing the tweets ...\\n"\nclean_tweets = []\nfor i in xrange(nums[0],nums[1]):\n    if((i+1)%5000 == 0):\n        print "tweets %d" %(i+1)\n    clean_tweets.append(tweet_cleaner(dataframe[\'text\'][i]))')

get_ipython().run_cell_magic('time', '', 'print "cleaning and parsing the tweets ...\\n"\nfor i in xrange(nums[2],nums[3]):\n    if((i+1)%10000 == 0):\n        print "tweets %d" %(i+1)\n    clean_tweets.append(tweet_cleaner(dataframe[\'text\'][i]))')

len(clean_tweets)

#dividing and  saving cleaned data to csv
clean_dataframe_neg = pd.DataFrame(clean_tweets[:40000],columns=['text'])
clean_dataframe_neg['sentiment'] = 0

clean_dataframe_pos = pd.DataFrame(clean_tweets[40000:80000],columns=['text'])
clean_dataframe_pos['sentiment'] = 4

clean_dataframe_pos.to_csv('clean_tweet_pos.csv',encoding='utf-8')

clean_dataframe_neg.to_csv('clean_tweet_neg.csv',encoding='utf-8')

df_pos = pd.read_csv("clean_tweet_pos.csv",index_col=0)
df_pos['sentiment'] = df_pos['sentiment'].map({4: 1})
#df_pos.drop(['Unnamed: 0'])
df_pos.head()
df_neg = pd.read_csv("clean_tweet_neg.csv",index_col=0)
df_neg['sentiment'] = df_neg['sentiment'].map({0: 0})
df_neg.head()

df_pos.info()

df_pos[df_pos.isnull().any(axis=1)].head()

np.sum(df_pos.isnull().any(axis=1))

# we need to remove the rows which contain null text ( no use)
# reasons for null: either contain only character like html tags, url links, mention, www. 
# removing row containing any data null
df_pos.dropna(inplace=True)
df_pos.reset_index(drop=True,inplace=True)
df_pos.info()

df_neg.dropna(inplace=True)
df_neg.reset_index(drop=True,inplace=True)
df_neg.info()

neg_string = []
demo_string = []
for t in df_neg.text:
    neg_string.append(t)

demo_string = neg_string[:15]
neg_string = pd.Series(neg_string).str.cat(sep=' ')

demo_string = pd.Series(demo_string)
demo_string[:5]

demo_string = pd.Series(demo_string).str
demo_string[:5]

from wordcloud import WordCloud 

wordcloud_neg = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.show()

pos_string = []
for t in df_pos.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')

wordcloud_pos = WordCloud(width=1600, height=800,colormap='magma', max_font_size=200,background_color='white').generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud_pos, interpolation = "bilinear")
plt.axis("off")
plt.show()

def create_word_features(words):
    my_dict=dict([(word,True) for word in words])
    return my_dict

neg_vec=[]
for string in df_neg.text:
    words=nltk.word_tokenize(string.decode('utf-8'))
    neg_vec.append((create_word_features(words),'negative'))

pos_vec=[]
for string in df_pos.text:
    words=nltk.word_tokenize(string.decode('utf-8'))
    pos_vec.append((create_word_features(words),'positive'))

neg_vec

train_set=neg_vec+pos_vec

classifier=NaiveBayesClassifier.train(train_set)

cols = ['sentiment','id','date','query_string','user','text']
df_test = pd.read_csv("test_data.csv",header=None, names=cols,encoding=None)
df_test.head()

df_test.drop(['id','date','query_string','user'],axis=1,inplace=True)
df_test.sentiment.value_counts()

negtweets_test=(df_test[df_test.sentiment == 0].text).tolist()
neutweets_test=(df_test[df_test.sentiment == 2].text).tolist() #length is 0
postweets_test=(df_test[df_test.sentiment == 4].text).tolist()

neg_tweet_test=[]
for string in negtweets_test:
    words=nltk.word_tokenize(string.decode('utf-8'))
    neg_tweet_test.append((create_word_features(words),'negative'))

pos_tweet_test=[]
for string in postweets_test:
    words=nltk.word_tokenize(string.decode('utf-8'))
    pos_tweet_test.append((create_word_features(words),'positive'))

test_set=pos_tweet_test+neg_tweet_test

accuracy=nltk.classify.util.accuracy(classifier,test_set)
print accuracy*100

