from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

app = pd.read_pickle('app_cleaned.pickle')

app = app.drop_duplicates()

app = app.dropna(axis = 0)#remove the NAN

app.head()

ratio = app['num_current_rating']/app['num_overall_rating']

#use histogram to show the range of ratio
plt.hist(ratio,bins = 20, alpha = .4, label = 'ratio')
plt.legend()
plt.show()

index = ratio>0.05#get the index of ratio larger than 0.05

appfilter = app.loc[index]#filter the apps which number of current rating over number of overall rating larger than 0.1

#use histogram to show the range of current_rating-overall_rating
plt.hist(appfilter['current_rating']-appfilter['overall_rating'],bins = 20, alpha = .4, label = 'diff')
plt.legend()
plt.show()

diff = appfilter['current_rating']-appfilter['overall_rating']

index2 = diff>=0.1#get the index of the difference larger than 0.1
index2b = diff<= -0.1#get the index of the difference smaller than -0.1

appinprove = appfilter.loc[index2]
appdecrease = appfilter.loc[index2b]

nvd = appinprove['new_version_desc']
nvdd = appdecrease['new_version_desc']

#compile documents
doc_complete = nvd.tolist()
doc_complete2 = nvdd.tolist()

#clean doc
import nltk
from nltk import corpus
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
stemmer = PorterStemmer().stem
tokenize = nltk.word_tokenize
stop = stopwords.words('english')+list(string.punctuation)+['we','new','fix','io','updat','improv','bug',
                                                            'app','featur','perform','ad',"\'s","--","us"
                                                            ,"minor","support","iphon","issu","add","enhanc",
                                                           "user","pleas","10","7","experi","thank",
                                                           "version","experi","screen","\'\'","2","6","icon",
                                                           "stabil","review","5","``"]
def stem(tokens,stemmer = PorterStemmer().stem):
    stemwords = [stemmer(w.lower()) for w in tokens if w not in stop]
    return [w for w in stemwords if w not in stop]
def lemmatize(text):
    return stem(tokenize(text))

doc_clean = [lemmatize(doc) for doc in doc_complete]
doc_clean2 = [lemmatize(doc) for doc in doc_complete2]

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)
dictionary2 = corpora.Dictionary(doc_clean2)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
doc_term_matrix2 = [dictionary2.doc2bow(doc) for doc in doc_clean2]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
ldamodel2 = Lda(doc_term_matrix2, num_topics=3, id2word = dictionary2, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))
print(ldamodel2.print_topics(num_topics=3, num_words=3))

index_interfac = []
for i in range(len(doc_clean)):
    if 'interfac' in doc_clean[i]:
        index_interfac.append(True)
    else:
        index_interfac.append(False)

nvd[index_interfac][1342]

index_feedback = []
for i in range(len(doc_clean)):
    if 'feedback' in doc_clean[i]:
        index_feedback.append(True)
    else:
        index_feedback.append(False)

nvd[index_feedback][193]

index_store = []
for i in range(len(doc_clean)):
    if 'store' in doc_clean[i]:
        index_store.append(True)
    else:
        index_store.append(False)

nvd[index_store][1024]

index_ipad = []
for i in range(len(doc_clean2)):
    if 'ipad' in doc_clean2[i]:
        index_ipad.append(True)
    else:
        index_ipad.append(False)

nvdd[index_ipad][1373]

index_music = []
for i in range(len(doc_clean2)):
    if 'music' in doc_clean2[i]:
        index_music.append(True)
    else:
        index_music.append(False)

nvdd[index_music][2157]

index_card = []
for i in range(len(doc_clean2)):
    if 'card' in doc_clean2[i]:
        index_card.append(True)
    else:
        index_card.append(False)

nvdd[index_card][646]

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
dec_improv = pyLDAvis.gensim.prepare(ldamodel,doc_term_matrix, dictionary)
dec_decrea = pyLDAvis.gensim.prepare(ldamodel2,doc_term_matrix2, dictionary2)

dec_improv

pyLDAvis.save_html(dec_improv,'improved_apps.html')

dec_decrea

pyLDAvis.save_html(dec_decrea,'worsen_apps.html')



