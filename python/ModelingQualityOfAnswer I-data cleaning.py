import sqlite3
import pickle
import pandas as pd

conn = sqlite3.connect('../Data/crossvalidated.db')

# return all the records for questions posts from posts table
ques_query = "SELECT * FROM [posts] WHERE PostTypeId==2"

apost_df = pd.read_sql_query(ques_query, conn)

apost_df.shape

apost_df.columns

import multiprocessing
multiprocessing.cpu_count()

apost_df.drop(['LastEditorDisplayName','CommunityOwnedDate','LastEditorUserId','LastEditDate',
             'LastActivityDate'],axis=1,inplace=True)

#no closed date for answer
apost_df[apost_df.ClosedDate.isnull()].shape

bins = [-36, -1, 0, 2, 15, 260]
group_names = ['bad','neutral','satisfactory','good','awesome']
apost_df['AnsQuality']= pd.cut(apost_df['Score'],bins,labels=group_names)

apost_df.AnsQuality.value_counts()

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from textblob import TextBlob, Word
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import re
import string

stop_words = pd.read_csv("../Data/stoplist copy.csv",header=None)

stop_words = stop_words[0].tolist()

stop_words = set(stop_words + stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))

with open('../Data/stop_words.pickle', 'rb') as handle:
  stop_words = pickle.load(handle)

#still may want to hand craft a little bit
stop_words = stop_words.union(set(['don','le', 'isthe', 'likeif','ll','ve','cohen','se','setof','isn']))

#be careful , like p-value, t-distribution
stop_words = stop_words - set(['p','t'])

import pickle
with open('../Data/stop_words.pickle', 'wb') as handle:
    pickle.dump(stop_words, handle)

verb_exp = set(['VB', 'VBZ', 'VBP', 'VBD','VBN','VBG'])
#porter_stemmer = PorterStemmer()
def clean_text(row):
    soup = BeautifulSoup(row, 'html.parser')
    #remove code
    for tag in soup.find_all('code'):
        tag.replaceWith(' ')
        
    raw = soup.get_text().lower()
    #remove link
    raw_no_link = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', raw)
    #remove email
    no_link_email = re.sub(r'[\w\.-]+@[\w\.-]+[\.][com|org|ch|uk]{2,3}', " ", raw_no_link)
    #remove whitespace
    tab_text = '\t\n\r\x0b\x0c'
    no_link_email_space = "".join([ch for ch in no_link_email if ch not in set(tab_text)])
    #remove fomula
    reg = '(\$.+?\$)|((\\\\begin\{.+?\})(.+?)(\\\\end\{(.+?)\}))'
    raw = re.sub(reg, " ", no_link_email_space, flags=re.IGNORECASE)   
    #remove numbers
    raw = re.sub('[0-9]+?', ' ', raw) 
    # remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    raw = regex.sub(' ', raw)
    #clean out the characters left out after the above step, like we’re, I’m, It’s, i.e., isn't
    raw = re.sub('( s )|( re )|( m )|( i e )|(n t )',' ',raw) 
    
    # lementize
    row_t = TextBlob(raw)
    raw = []
    for word, pos in row_t.tags:
        if pos in verb_exp:
            word = Word(word)
            word = word.lemmatize("v")
        else:
            word = Word(word)
            word = word.lemmatize()
        raw.append(word)
    clean = ' '.join(raw)   
    
    # remove stop words
    cleaned_text = " ".join([word for word in word_tokenize(clean) if word not in stop_words]) 
     
    return(cleaned_text)

apost_df['Body'][0]

clean_text(apost_df['Body'][0])

# get the cleaned body by removing stopwords and punctuation 
body_clean_sto_pun = apost_df['Body'].map(lambda i: clean_text(i))

body_clean_sto_pun[6]

type(body_clean_sto_pun)

import pickle
with open('../Data/ans_clean_text.pickle', 'wb') as handle:
    pickle.dump(body_clean_sto_pun, handle)

ans_quality = apost_df['AnsQuality'] 

import pickle
with open('../Data/ans_quality.pickle', 'wb') as handle:
    pickle.dump(ans_quality, handle)

tt = 't1;a, 48 t4. Ab! B2?fs/fdsa--fw'
tt.split()



