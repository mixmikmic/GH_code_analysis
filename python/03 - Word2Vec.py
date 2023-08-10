import pandas as pd
import pickle
import numpy as np

# Load the bar review dataset 
review = pd.read_pickle('../output/bar_restaurant_reviews_cleaned_and_tokenized.pickle')
review.head(5)


# Now let's generate a word2vec trained model on the dataset.
# First we need to override the simple weighting scheme



import gensim
from itertools import chain
import sys
sys.path.append('../vectorsearch/')
import nltk_helper
import word2vec



def create_vector_model(model, tokenized_docs, **kwargs):
    """
    Create gensim Word2Vec model out of review list
    where each element contains review
    """
    review_flatten = list(chain.from_iterable(tokenized_docs))
    print 'training word2vec model...'
    vec_model = model(review_flatten, **kwargs)
    return vec_model



# Arguments to the word2vec model
model_args = {'size':200, 'window':5, 'min_count':5, 'workers':12, 'iter':10}


word2vec_model = create_vector_model(model=word2vec.Word2Vec, 
                                     tokenized_docs=review.cleaned_tokenized.iloc[:],
                                     **model_args)
# Done training, so this reduces ram footprint.
word2vec_model.init_sims(replace=True)



word2vec_model.save('../output/word2vec_bars_and_restaurants.model')

import sys
sys.path.append('../vectorsearch/')
import nltk_helper
import word2vec

word2vec_model = word2vec.Word2Vec.load('../output/word2vec_bars_and_restaurants.model')

from query import parse_query
query_dict = parse_query('steak:1; ocean:2 ; land:1' )


for word, sim in word2vec_model.most_similar(query_dict, topn=20):
    print np.dot(word2vec_model[word], word2vec_model[word])


#print word2vec_model.most_similar({'pepperoni':10, 'cheese':1000000})










# model_args = {'num_topics':100}
# lda_model = create_vector_model(model=gensim.models.LdaModel, review_list=yelp_review_sample, **model_args)

#model.similarity('bar')
#model.most_similar('bar', topn=20)

# from gensim import corpora, models, similarities
# model = models.ldamodel.LdaModel(yelp_review_sample, num_topics=10)







from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')


from time import time
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(yelp_review_sample)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))


print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

lda.fit(yelp_review_sample)


import calculator
reload(calculator)

calc = calculator.word2vec_calc(word2vec_model)
calc.calc('dog+dog')



