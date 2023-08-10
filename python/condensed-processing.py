import pandas as pd
import numpy as np

#nlp
import re
from textblob import TextBlob
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from nltk.corpus import stopwords
stop = stopwords.words('english')

import warnings
warnings.filterwarnings('ignore')

def add_datepart(df, fldname, drop=False, time=True):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear']
    if time: attr = attr + ['Hour', 'Minute']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

#read in the data
tweets = pd.read_csv('../../finalprojdata/tweets.csv')
users = pd.read_csv('../../finalprojdata/users.csv')

users = users.drop_duplicates('id')
tweets.drop(['tweet_id','retweeted_status_id', 'in_reply_to_status_id', 'created_at', 'expanded_urls'], axis=1, inplace=True)

fulltweets = tweets.merge(users, how='left', left_on='user_id', right_on='id')
fulltweets = fulltweets[pd.notnull(fulltweets['user_id'])]
fulltweets = fulltweets[pd.notnull(fulltweets['created_str'])]
fulltweets = fulltweets[pd.notnull(fulltweets['friends_count'])]
fulltweets = fulltweets[pd.notnull(fulltweets['time_zone'])]

#fulltweets.isnull().sum()

#fix time
add_datepart(fulltweets, 'created_str')

#get rid of special characters
tweetsonly2 = fulltweets.text.copy().astype(str)
tweetsonly2 = tweetsonly2.str.replace('[^\w\s]','')
tweetsonly2 = tweetsonly2.str.replace('[\\r|\\n|\\t|_]',' ')
tweetsonly2 = tweetsonly2.str.strip()

fulltweets2 = fulltweets.copy()
fulltweets2.text = tweetsonly2

## Create a sentiment column
def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

## Create a subjectivity column
def analyze_sentiment2(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tweet)
    return analysis.sentiment.subjectivity

# create sentiment and subjectivity columns
stop += ['rt']
fulltweets2.text = fulltweets2.text.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))
fulltweets2['Sentiment'] = np.array([analyze_sentiment(str(tweet)) for tweet in fulltweets2.text.values])
fulltweets2['Subjectivity'] = np.array([analyze_sentiment2(str(tweet)) for tweet in fulltweets2.text.values])
fulltweets2.text = fulltweets2.text.apply(lambda x: ' '.join([word.lower() for word in x.split() if len(word) > 3]))

fulltweets2.to_csv('../../finalprojdata/fulltweets2.csv')

