import pandas as pd
import pymysql
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
import requests
import numpy as np

# Fetch the data from the mysql server and throw it into a dataframe
engine = create_engine('mysql+pymysql://newsfuzz:newsfuzzplease@newsfuzz.cuhvcgseshha.eu-west-2.rds.amazonaws.com:3306/newsfuzz', encoding='utf-8')
newsfuzz_db = pd.io.sql.read_sql('SELECT * FROM newsfuzz_db_test', engine, index_col='index')

print(len(newsfuzz_db))
newsfuzz_db.head()

np.unique(newsfuzz_db['source_id'])

# Print out the url of the first story for each news source
for source in np.unique(newsfuzz_db['source_id']):
    print(newsfuzz_db[newsfuzz_db['source_id'].isin([source])]['article_url'].tolist()[1])

def get_soup(url):
    res = requests.get(url)
    res.raise_for_status()
    return BeautifulSoup(res.text,'lxml')

def parse_news(url):
    # get soup
    try:
        s= get_soup(url)
        for script in s(["script", "style"]):
            script.extract()
        text=''
        # determine which parser to use
        if 'dailymail.co.uk' in url:
            text=s.find('div',attrs={"itemprop":"articleBody"}).text.replace('\n', '')
        if 'bbc.co.uk/news' in url:
            text=s.find('div',attrs={"class":"story-body__inner"}).text.replace('\n', '')
        if 'abc.net.au' in url:
            text=s.find('div',attrs={"class":"article section"}).text.replace('\n', '')
        if 'theguardian.com' in url:
            text=s.find('div',attrs={"itemprop":"articleBody"}).text.replace('\n', '')
        if 'telegraph.co.uk' in url:
            text=s.find('div',attrs={"class":"article__content js-article"}).text.replace('\n', '')
        if 'mirror.co.uk' in url:
            text=s.find('div',attrs={"class":"article-body"}).text.replace('\n', '')
        if 'reuters.com' in url:
            text=s.find('span',attrs={"id":"article-text"}).text.replace('\n', '')   
        if 'breitbart.com' in url:
            text=s.find('div',attrs={"class":"entry-content"}).text.replace('\n', '')  
    except Exception as exc:
            print(exc)
    return text

parse_news('http://www.breitbart.com/big-government/2017/06/11/donald-trump-ridicules-cowardly-leaker-james-comey/')

# articles=newsfuzz_db[newsfuzz_db['source_id'].isin(['the-guardian-uk','daily-mail','bbc-news','abc-news-au','mirror','the-telegraph'])]
articles_gdn=newsfuzz_db[newsfuzz_db['source_id'].isin(['the-guardian-uk'])]

articles_gdn.head()

gdn_urls=articles_gdn['article_url'].tolist()

def get_articles(url_list):
    articles_text=[]
    count=0
    for u in urls:
        count=count+1
        if count % 10 == 0:
            print(str(count)+' articles from a total of '+str(len(url_list)))
        articles_text.append(parse_news(u))
    return articles_text

articles_text=get_articles(gdn_urls)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords 

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# documents = articles_guard_dm
documents=articles_text
stop = set(stopwords.words('english'))
no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stop)
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stop)
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 10

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=50, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
#display_topics(lda, tf_feature_names, no_top_words)



