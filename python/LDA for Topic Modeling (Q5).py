from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import nltk
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
import lda

app = pd.read_pickle('app_cleaned.pickle')

app.columns

app['weighted_rating'] = map(lambda a, b, c,d: np.divide(a,b)*c+(1-np.divide(a,b))*d, app['num_current_rating'], 
                                  app['num_overall_rating'], app['current_rating'], app['overall_rating'])

good_app = app.loc[app['weighted_rating'] >=4.0]
bad_app = app.loc[app['weighted_rating'] <=2.5]
good_app = good_app.reset_index(drop=True)
bad_app = bad_app.reset_index(drop=True)

category = app['category']
cate_list = []
for i in category.unique():
    cate = i.lower()
    cate_list.append(cate)

first_good= good_app.loc[good_app['review1_star']>=4].reset_index(drop=True)['review1']
second_good = good_app.loc[good_app['review2_star']>=4].reset_index(drop=True)['review2']
third_good = good_app.loc[good_app['review3_star']>=4].reset_index(drop=True)['review3']
first_bad = bad_app.loc[bad_app['review1_star']<=2.5].reset_index(drop=True)['review1']
second_bad = bad_app.loc[bad_app['review2_star']<=2.5].reset_index(drop=True)['review2']
third_bad = bad_app.loc[bad_app['review3_star']<=2.5].reset_index(drop=True)['review3']

good_rev = first_good.append(second_good)
all_good = good_rev.append(third_good)
bad_rev = first_bad.append(second_bad)
all_bad = bad_rev.append(third_bad)

stop = set(stopwords.words('english')+[u'one',u'app',u'it',u'dont',u"i",u"'s","''","``",u'use',u'used',u'using',u'love',
                                      u'would',u'great',u'app.',u'like',u'lot']+ cate_list)
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def stem(tokens,stemmer = PorterStemmer().stem):
    return [stemmer(w.lower()) for w in tokens if w not in stop]

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    tokenize = nltk.word_tokenize
    to_token = stem(tokenize(normalized))
    tags = nltk.pos_tag(to_token)
    dt_tags = [t for t in tags if t[1] in ["DT", "MD", "VBP","IN", "JJ","VB"]]
    for tag in dt_tags:
        normalized = " ".join(tok for tok in to_token if tok not in tag[0])
    return normalized

doc_clean_g1 = [clean(doc).split() for doc in first_good]
doc_clean_g2 = [clean(doc).split() for doc in second_good]
doc_clean_g3 = [clean(doc).split() for doc in third_good]
doc_clean_b1 = [clean(doc).split() for doc in first_bad]
doc_clean_b2 = [clean(doc).split() for doc in second_bad]
doc_clean_b3 = [clean(doc).split() for doc in third_bad]

doc_clean_good = [clean(doc).split() for doc in all_good]
doc_clean_bad = [clean(doc).split() for doc in all_bad]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary_g1 = corpora.Dictionary(doc_clean_g1)
dictionary_g2 = corpora.Dictionary(doc_clean_g2)
dictionary_g3 = corpora.Dictionary(doc_clean_g3)
dictionary_b1 = corpora.Dictionary(doc_clean_b1)
dictionary_b2 = corpora.Dictionary(doc_clean_b2)
dictionary_b3 = corpora.Dictionary(doc_clean_b3)

dictionary_good = corpora.Dictionary(doc_clean_good)
dictionary_bad = corpora.Dictionary(doc_clean_bad)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix_g1 = [dictionary_g1.doc2bow(doc) for doc in doc_clean_g1]
doc_term_matrix_g2 = [dictionary_g2.doc2bow(doc) for doc in doc_clean_g2]
doc_term_matrix_g3 = [dictionary_g3.doc2bow(doc) for doc in doc_clean_g3]
doc_term_matrix_b1 = [dictionary_b1.doc2bow(doc) for doc in doc_clean_b1]
doc_term_matrix_b2 = [dictionary_b2.doc2bow(doc) for doc in doc_clean_b2]
doc_term_matrix_b3 = [dictionary_b3.doc2bow(doc) for doc in doc_clean_b3]

doc_term_matrix_good = [dictionary_good.doc2bow(doc) for doc in doc_clean_good]
doc_term_matrix_bad = [dictionary_bad.doc2bow(doc) for doc in doc_clean_bad]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel_g1 = Lda(doc_term_matrix_g1, num_topics=3, id2word = dictionary_g1, passes=50)
ldamodel_g2 = Lda(doc_term_matrix_g2, num_topics=3, id2word = dictionary_g2, passes=50)
ldamodel_g3 = Lda(doc_term_matrix_g3, num_topics=3, id2word = dictionary_g3, passes=50)
ldamodel_b1 = Lda(doc_term_matrix_b1, num_topics=3, id2word = dictionary_b1, passes=50)
ldamodel_b2 = Lda(doc_term_matrix_b2, num_topics=3, id2word = dictionary_b2, passes=50)
ldamodel_b3 = Lda(doc_term_matrix_b3, num_topics=3, id2word = dictionary_b3, passes=50)

print(ldamodel_g1.print_topics(num_topics=3, num_words=5))
print(ldamodel_g2.print_topics(num_topics=3, num_words=5))
print(ldamodel_g3.print_topics(num_topics=3, num_words=5))

print(ldamodel_b1.print_topics(num_topics=3, num_words=5))
print(ldamodel_b2.print_topics(num_topics=3, num_words=5))
print(ldamodel_b3.print_topics(num_topics=3, num_words=5))

ldamodel_good = Lda(doc_term_matrix_good, num_topics=10, id2word = dictionary_good, passes=20)
ldamodel_bad = Lda(doc_term_matrix_bad, num_topics=10, id2word = dictionary_bad, passes=20)

print(ldamodel_good.print_topics(num_topics=5, num_words=3))
print(ldamodel_bad.print_topics(num_topics=5, num_words=3))

import pyLDAvis
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
good_rev = pyLDAvis.gensim.prepare(ldamodel_good, doc_term_matrix_good, dictionary_good)
bad_rev = pyLDAvis.gensim.prepare(ldamodel_bad, doc_term_matrix_bad, dictionary_bad)

pyLDAvis.save_html(good_rev,"good_rev.html")
good_rev

bad_rev
pyLDAvis.save_html(bad_rev,"bad_rev.html")
bad_rev



