from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

get_ipython().magic('pylab inline')

import pandas as pd
import numpy as np
import warnings
from multiprocessing.dummy import Pool as ThreadPool 
import dateutil.parser
import nltk
import os
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix 
from multiprocessing import Pool
from nltk import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from nltk.stem.lancaster import LancasterStemmer

warnings.filterwarnings('ignore')

df = pd.read_csv("Complete_data.csv")

df.info()


x = list(range(0,len(df)-1))
for i in x:
    if "html" in str(df["Content"][i]):
        df.drop([i], inplace = True)

df.drop("Unnamed: 0",1,inplace = True)
df.fillna("Not Available", inplace = True)

#convert the dates in the emails to a datetime object
def to_date(datestring):
    date = dateutil.parser.parse(datestring)
    return date

df["Date"] = df["Date"].apply(to_date)

count = 0
for i in df["Content"]:
    if "html" in i:
        df = df.drop(df.index[count])
    count += 1

df["Content"] = df["Content"].apply(lambda x: x.replace("http", ''))
df["Content"] = df["Content"].apply(lambda x: x.replace(".com", ''))
df["Content"] = df["Content"].apply(lambda x: x.replace("www", ''))

common_html = ["blockquote", 'body', 'center', "del", 'div', 'font', 'head', ' hr ', 'block', 'align', '0px', '3d', 'arial', 'background', 'bgcolor', ' br ', 'cellpadding', 'cellspacing',
              'div', 'font', 'height', 'helvetica','href', 'img', 'valign', 'width', 'strong', 'serif', 'sans', ' alt ', 'display', 'src', 'style', ' tr ', 'tdtable', ' td ', 'tdtr', ' ef '
              'png', 'text', ' id ', 'gov', 'net']

for i in common_html:
    df["Content"] = df["Content"].apply(lambda x: x.replace(i, ''))

from nltk.stem.wordnet import WordNetLemmatizer #To download corpora: python -m    nltk.downloader all
lmtzr=WordNetLemmatizer()
df["Content"] = df["Content"].apply(lambda x: " ".join(lmtzr.lemmatize(word) for word in x.split(" ")))

text = []
for i in df["Content"]:
    text.append(i)

import pickle 

with open('text.pkl', 'wb') as f:
    pickle.dump(text,f)

len(text)

count = 0
for i in text:
    for word in i:
        count += 1
count

vectorizer = CountVectorizer(min_df = 50, max_df = .1,stop_words = 'english')
dtm = vectorizer.fit_transform(text)

from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities, matutils 

doc_vecs = vectorizer.transform(text).transpose()
doc_vecs.shape

corpus = matutils.Sparse2Corpus(doc_vecs)

id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

len(id2word)

lda = models.LdaMulticore(corpus, id2word=id2word, num_topics=20, passes=20)

lda.print_topics(num_words=10, num_topics=20)

sorted(lda.show_topic(10, topn=10), key=lambda x: x[1], reverse=True) [:10]

sorted(lda.show_topic(1, topn=10), key=lambda x: x[1], reverse=True) [:10]

# Transform the docs from the word space to the topic space (like "transform" in sklearn)
lda_corpus = lda[corpus]

lda_docs = [doc for doc in lda_corpus]

len(lda_docs)

np.save('lda.npy', lda)

np.save('lda_docs.npy',lda_docs)

topics_matrix = lda.show_topics(formatted=False, num_words=10, num_topics = 20)

len(topics_matrix)

topics = []
for i in topics_matrix:
    topics.append(i[1])

topics
topics_only = []
for i in topics:
    next_topic = []
    for element in i:
        next_topic.append(element[0])
    topics_only.append(next_topic)

for i in topics_only:
    print (i)

