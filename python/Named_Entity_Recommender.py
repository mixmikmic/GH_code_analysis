PATH_NEWS_ARTICLES="/home/phoenix/Documents/HandsOn/Final/news_articles.csv"
ARTICLES_READ=[4,5,7,8]
NUM_RECOMMENDED_ARTICLES=5
ALPHA = 0.5

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem.snowball import SnowballStemmer
import nltk
import numpy
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
stemmer = SnowballStemmer("english")

news_articles = pd.read_csv(PATH_NEWS_ARTICLES)
news_articles.head()

#Select relevant columns and remove rows with missing values
news_articles = news_articles[['Article_Id','Title','Content']].dropna()
articles = news_articles['Content'].tolist()
articles[0] #an uncleaned article

def clean_tokenize(document):
    document = re.sub('[^\w_\s-]', ' ',document)                           #remove punctuation marks and other symbols
    tokens = nltk.word_tokenize(document)                                  #Tokenize sentences
    cleaned_article = ' '.join([stemmer.stem(item) for item in tokens])    #Stemming each token
    return cleaned_article

cleaned_articles = map(clean_tokenize, articles)
cleaned_articles[0]  #a cleaned, tokenized and stemmed article 

#Generate tfidf matrix model 
tfidf_matrix = TfidfVectorizer(stop_words='english', min_df=2)
articles_tfidf_matrix = tfidf_matrix.fit_transform(cleaned_articles)
articles_tfidf_matrix #tfidf vector of an article

def get_ner(article):
    ne_tree = ne_chunk(pos_tag(word_tokenize(article)))
    iob_tagged = tree2conlltags(ne_tree)
    ner_token = ' '.join([token for token,pos,ner_tag in iob_tagged if not ner_tag==u'O']) #Discarding tokens with 'Other' tag
    return ner_token

#Represent user in terms of cleaned content of read articles
user_articles = ' '.join(cleaned_articles[i] for i in ARTICLES_READ) 
print "User Article =>", user_articles
print '\n'
#Represent user in terms of NERs assciated with read articles 
user_articles_ner = ' '.join([get_ner(articles[i]) for i in ARTICLES_READ])
print "NERs of Read Article =>", user_articles_ner

#Get vector representation for both of the user read article representation
user_articles_tfidf_vector = tfidf_matrix.transform([user_articles])
user_articles_ner_tfidf_vector = tfidf_matrix.transform([user_articles_ner])
user_articles_tfidf_vector

# User_Vector =>  (Alpha) [TF-IDF Vector] + (1-Alpha) [NER Vector] 
alpha_tfidf_vector = ALPHA * user_articles_tfidf_vector
alpha_ner_vector = (1-ALPHA) * user_articles_ner_tfidf_vector

user_vector = np.sum(zip(alpha_tfidf_vector,alpha_ner_vector))
user_vector

user_vector.toarray()

def calculate_cosine_similarity(articles_tfidf_matrix, user_vector):
    articles_similarity_score=cosine_similarity(articles_tfidf_matrix, user_vector.toarray())
    recommended_articles_id = articles_similarity_score.flatten().argsort()[::-1]
    #Remove read articles from recommendations
    final_recommended_articles_id = [article_id for article_id in recommended_articles_id 
                                     if article_id not in ARTICLES_READ ][:NUM_RECOMMENDED_ARTICLES]
    return final_recommended_articles_id

recommended_articles_id = calculate_cosine_similarity(articles_tfidf_matrix, user_vector)
recommended_articles_id

#Recommended Articles and their title
#df_news = pd.read_csv(PATH_NEWS_ARTICLES)
print 'Articles Read'
print news_articles.loc[news_articles['Article_Id'].isin(ARTICLES_READ)]['Title']
print '\n'
print 'Recommender '
print news_articles.loc[news_articles['Article_Id'].isin(recommended_articles_id)]['Title']

ALPHA = 0

alpha_tfidf_vector = ALPHA *user_articles_tfidf_vector    # ==> 0
alpha_ner_vector = (1-ALPHA) * user_articles_ner_tfidf_vector

user_vector = np.sum(zip(alpha_tfidf_vector,alpha_ner_vector))
user_vector

user_vector.toarray()

recommended_articles_id = calculate_cosine_similarity(articles_tfidf_matrix, user_vector)
recommended_articles_id

#Recommended Articles and their title
#df_news = pd.read_csv(PATH_NEWS_ARTICLES)
print 'Articles Read'
print news_articles.loc[news_articles['Article_Id'].isin(ARTICLES_READ)]['Title']
print '\n'
print 'Recommender '
print news_articles.loc[news_articles['Article_Id'].isin(recommended_articles_id)]['Title']



