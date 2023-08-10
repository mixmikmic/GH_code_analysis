import pandas as pd
import numpy as np

#from: http://help.sentiment140.com/for-students/

'''
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)
'''

s140_train = pd.read_csv('model_data/training.1600000.processed.noemoticon.csv', encoding='latin-1',
                         names=['sentiment','id','date','query','user','text'], header = None)

s140_train = s140_train[['sentiment','text']]

s140_train = s140_train[s140_train['sentiment'] != 2]


X_train = s140_train['text']
y_train = np.where(s140_train['sentiment'] == 0, 0, 1)

y_train.shape, X_train.shape

X_train[:20]

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
stopwords_filtered = list(stopwords_nltk.difference(relevant_words))

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                    stop_words = stopwords_filtered, max_features = 10000, ngram_range = (1,2))

words_matrix = vectorizer.fit_transform(X_train)
vocabulary = vectorizer.get_feature_names()

vocabulary[:20]

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression() 
logistic_model.fit(words_matrix, y_train)
vocabulary = vectorizer.get_feature_names()
coefs = logistic_model.coef_
word_importances = pd.DataFrame({'word': vocabulary, 'coef': coefs.tolist()[0]})
word_importances_sorted = word_importances.sort_values(by='coef', ascending = False)
print(word_importances_sorted)

from sklearn.externals import joblib
joblib.dump(logistic_model, 'logistic_model.pkl') 

joblib.dump(vectorizer, 'vectorizer.pkl')

