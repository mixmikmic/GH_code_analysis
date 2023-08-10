import pandas as pd 
from gensim import corpora, models, similarities

review = pd.read_pickle('../output/bar_reviews_cleaned_and_tokenized.pickle')

from itertools import chain
from collections import OrderedDict
reviews_merged = OrderedDict()

# Flatten the reviews, so each review is just a single list of words.


n_reviews = -1

for bus_id in set(review.business_id.values[:n_reviews]):
    # This horrible line first collapses each review of a corresponding business into a list
    # of lists, and then collapses the list of sentences to a long list of words
    reviews_merged[bus_id] = list(chain.from_iterable( 
                                    chain.from_iterable( review.cleaned_tokenized[review.business_id==bus_id] )))
    

import time 
from itertools import chain

print 'Generating vector dictionary....'
 # Review level LDA
# review_flatten = list(chain.from_iterable(review.cleaned_tokenized.iloc[:])) 
# id2word_wiki = corpora.Dictionary(review_flatten)


start = time.time()

# Business level LDA (all reviews for a business merged)
id2word_wiki = corpora.Dictionary(reviews_merged.values())

print 'Dictonary generated in %1.2f seconds'%(time.time()-start)

# Convert corpus to bag of words for use with gensim...
# See https://radimrehurek.com/gensim/tut1.html#from-strings-to-vectors
#corpus = map(lambda doc: id2word_wiki.doc2bow(doc), review_flatten)
corpus = map(lambda doc: id2word_wiki.doc2bow(doc), reviews_merged.values())
corpora.MmCorpus.serialize('../output/bar_corpus.mm', corpus)


# Can load the corpus with 
# from gensim import corpora
# corpus = corpora.MmCorpus('../output/bar_corpus.mm')



import gensim
print 'Fitting LDA Model'
start = time.time()
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, 
                                           id2word=id2word_wiki, passes=5,)
print 'LDA fit in %1.2f seconds'%(time.time()-start)

for topic in ldamodel.print_topics(num_topics=10, num_words=8): 
    print topic

from sklearn.decomposition import LatentDirichletAllocation, nmf


lda = LatentDirichletAllocation(n_topics=10, evaluate_every=1000, n_jobs=12, verbose=True)

lda.fit(corpus[:2000])



