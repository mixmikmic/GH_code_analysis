from sklearn.externals import joblib
logistic_model = joblib.load('logistic_model.pkl') 

vectorizer = joblib.load('vectorizer.pkl')

import numpy as np
import pandas as pd
import config
import re
from nltk.corpus import stopwords
from ast import literal_eval
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_colwidth', 200)

english_file = "{}/data/twitter/tweets/trivadis/2016-11_english.csv".format(config.dir_prefix)

tweets_english = pd.read_csv(english_file, encoding='utf-8', 
                              usecols = ['id_str', 'user_id', 'created_at', 'lang', 'text', 'favorite_count', 'entities',
                                         'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                                         'retweet_count', 'quoted_status_id_str', 'text_tokenized', 'text_processed'],
                              converters={"text_tokenized": literal_eval, "text_processed": literal_eval})

def remove_hash(wordlist):
    return(list(map(lambda x: re.sub(r'^#','',x), wordlist)))

def remove_at(wordlist):
    return(list(map(lambda x: re.sub(r'^@','',x), wordlist)))
    
tweets_english['text_wo_#'] = tweets_english['text_processed'].apply(lambda x: remove_hash(x))
tweets_english['text_wo_#@'] = tweets_english['text_wo_#'].apply(lambda x: remove_at(x))

X_train_en = tweets_english['text_wo_#'].apply(lambda x: ' '.join(x))

stopwords_nltk = set(stopwords.words("english"))
#relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
relevant_words = set()
additional_stopwords = set(['us'])
stopwords_filtered = list(additional_stopwords.union(stopwords_nltk.difference(relevant_words)))

words_matrix = vectorizer.transform(X_train_en)
words_matrix

y_pred = logistic_model.predict(words_matrix)
y_pred_prob = logistic_model.predict_proba(words_matrix)

len(X_train_en), len(y_pred)

y_pred[:10], y_pred_prob[:10]

y_pred_prob[:,1]

X_labeled = pd.DataFrame({'text': X_train_en, 'sentiment': y_pred, 'prob': y_pred_prob[:,1]})

X_labeled.head()

X_labeled.describe()

X_labeled[X_labeled['sentiment'] == 0].drop_duplicates()

new_vectorizer = CountVectorizer(analyzer = "word", tokenizer = str.split, 
                                    stop_words = stopwords_filtered, max_features = 100000, ngram_range = (1,1))
new_matrix = new_vectorizer.fit_transform(X_labeled['text'])
new_feature_names = new_vectorizer.get_feature_names()
new_feature_names[:50]

y = X_labeled['sentiment']

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression() 
logistic_model.fit(new_matrix, y)
vocabulary = new_vectorizer.get_feature_names()
coefs = logistic_model.coef_
word_importances = pd.DataFrame({'word': vocabulary, 'coef': coefs.tolist()[0]})
word_importances_sorted = word_importances.sort_values(by='coef', ascending = False)
print(word_importances_sorted)



