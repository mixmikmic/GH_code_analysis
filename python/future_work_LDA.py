#import libraries and set seaborn styling
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tmdbsimple as tmdb
import requests
import pandas as pd
import time
import numpy as np
from ast import literal_eval
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
sns.set_context('talk')
sns.set_style('ticks')

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition.online_lda import LatentDirichletAllocation

import matplotlib.pyplot as plt
import seaborn as sns

import pyLDAvis
import pyLDAvis.sklearn

movies = pd.read_csv('data/movies.csv')

#define tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#set stop words list
english_stop = get_stop_words('en')
print(len(english_stop))

#function to clean plots
def clean_plot(plot):
    '''
    clean_plot()
    -applies the following the plot of a movie:
        1) lowers all strings
        2) tokenizes each word
        3) removed English stop words

    -inputs: plot (string)
    
    -outputs: list representation of plot
    '''
    plot = plot.lower()
    plot = tokenizer.tokenize(plot)
    plot = [word for word in plot if word not in english_stop]
    return plot

#apply to movies df for both imdb and tmdb
movies['tmdb_clean_plot'] = movies['tmdb_plot'].apply(lambda x: clean_plot(x))
movies['imdb_clean_plot'] = movies['imdb_plot'].apply(lambda x: clean_plot(x))
movies['combined_clean_plot'] = movies['combined_plots'].apply(lambda x: clean_plot(x))

movies.head(2)

def post_process(list1):
    str1 = " ".join(list1)
    return str1

movies['post_tmdb_clean_plot'] = movies['tmdb_clean_plot'].apply(lambda x: post_process(x))

movies.post_combined_clean_plot[0]

english_stop = get_stop_words('en')

tfidf_vectorizer  = TfidfVectorizer(max_features=8000,max_df=0.9,min_df=0.02,stop_words=english_stop,ngram_range=(1,10),lowercase=True)
  
tmdb_bow = tfidf_vectorizer.fit_transform(movies['post_tmdb_clean_plot'])

lda = LatentDirichletAllocation(n_components=19, max_iter=100,learning_offset=200,learning_method='online',random_state=10)

tmdb_bow_prob = lda.fit_transform(tmdb_bow)*100

tmdb_prob_data = pd.DataFrame(np.around(tmdb_bow_prob,2),index=movies.title)

#https://pandas.pydata.org/pandas-docs/stable/style.html
cm = sns.light_palette("lightblue", as_cmap=True)
datafram_colored = tmdb_prob_data.sample(n=5,random_state=5).style.background_gradient(cmap=cm)
datafram_colored

#http://pyldavis.readthedocs.io/en/latest/
pyLDAvis.enable_notebook()
py_data = pyLDAvis.sklearn.prepare(lda, tmdb_bow, tfidf_vectorizer)
pyLDAvis.display(py_data)



