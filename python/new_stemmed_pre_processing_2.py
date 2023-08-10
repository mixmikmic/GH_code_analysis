import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim import utils
import os
import nltk
import scipy.sparse as ssp
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

train=pd.read_csv("checkpoints_databases/new_working_train.csv",encoding="utf8")
test=pd.read_csv("checkpoints_databases/new_working_test.csv",encoding="utf8")

#Stock as input features for meta model
train_cl=train.drop(["Variation","Class","Gene","Full_Text","Window_Text"],axis=1)
test_cl=test.drop(["Class","Variation","Gene","Full_Text","Window_Text"],axis=1)
train_cl.to_csv("w_meta_features/meta_train_l1l2.csv",index=False)
test_cl.to_csv("w_meta_features/meta_test_l1l2.csv",index=False)

data_all=pd.concat((train,test)).reset_index(drop=True)

data_all["Window_Text"][data_all["Window_Text"].isnull()==True]="null"

stop=['a','about','above','after','again','ain','am', 'an','and','any','are','aren','as','at','be','because','been','before','being',
 'below','between','both','but','by','d','down','during','for','from','further','had','hadn','has','hasn','have','haven','having',
'he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn','it','its','itself','just','ll','m',
'ma','me','more','most','my','myself','needn','no','nor','not','now','o','of','off','on','once','only','or','other','our','ours',
 'ourselves','out','over','own','re','s','same','shan','she','so','some','such','t','than','that','the','their','theirs','them',
 'themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn','we',
 'were','weren','what','when','where','which','while','who','whom','why','will','with','y','you','your','yours','yourself','yourselves']

exclude = set('.,!"#$%&\'()*+:;<=>?@[\\]^_`{|}0123456789')
ps=PorterStemmer()
lemma=WordNetLemmatizer()
def clean(doc,lemmatiz=False,stemming=False):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free_0 =[re.sub(",|\.|/"," ",ch) for ch in stop_free]
    punc_free_lem="".join(ch for ch in punc_free_0 if ch not in exclude)
    if lemmatiz==True:
        lem=[]
        for word,tag in pos_tag(word_tokenize(punc_free_lem)):
            wntag=tag[0].lower()
            wntag=wntag if wntag in ["a","r","n","v"] else None
            if not wntag:
                lem.append(word)
            else:
                lem.append(lemma.lemmatize(word,wntag))
        normalized=" ".join(word for word in lem)
        return normalized
    if stemming==True:
        normalized=" ".join(ps.stem(word) for word in word_tokenize(punc_free_lem))
        return normalized
    else:
        return ("Choose a cleaning man")

data_all["Window_Text"] = [clean(doc,stemming=True) for doc in data_all["Window_Text"]]
data_all["Full_Text"] = [clean(doc,lemmatiz=True) for doc in data_all["Full_Text"]]

train = data_all.iloc[:len(train)]
test = data_all.iloc[len(train):]

class MySentences(object):
    """MySentences is a generator to produce a list of tokenized sentences 
    
    Takes a list of numpy arrays containing documents.
    
    Args:
        arrays: List of arrays, where each element in the array contains a document.
    """
    def __init__(self, *arrays):
        self.arrays = arrays
 
    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)

def get_word2vec(sentences, location,size):
    """Returns trained word2vec
    
    Args:
        sentences: iterator for sentences
        
        location (str): Path to save/load word2vec
    """
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model
    
    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=size, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model

#It's important to remove duplicated spaces for word2vec learning !
train["Full_Text"]=[" ".join(doc.split()) for doc in train["Full_Text"].values]
test["Full_Text"]=[" ".join(doc.split()) for doc in test["Full_Text"].values]
train["Window_Text"]=[" ".join(doc.split()) for doc in train["Window_Text"].values]
test["Window_Text"]=[" ".join(doc.split()) for doc in test["Window_Text"].values]

number_w2v=[300] # we know it's 300 from previous runs, no time to gridsearch again and fit weights for lowers 
w2v={}
for size in number_w2v:
    w2v["w2v_"+str(size)] = get_word2vec(
        MySentences(
            train["Window_Text"].values),"new_stem_w2v_features"+str(size),size
    )

class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

mean_embedding_vectorizer={}
mean_embedded_train={}
mean_embedded_test={}
for name in w2v:
    mean_embedding_vectorizer[name] = MeanEmbeddingVectorizer(w2v[name])
    mean_embedded_train[name] = mean_embedding_vectorizer[name].fit_transform(train['Window_Text'])
    mean_embedded_test[name] = mean_embedding_vectorizer[name].fit_transform(test['Window_Text'])
df_embed_tr={}
df_embed_te={}
for name in w2v:
    df_embed_tr[name]=pd.DataFrame(mean_embedded_train[name])
    df_embed_te[name]=pd.DataFrame(mean_embedded_test[name])
train_w2v={}
test_w2v={}
for name in w2v:
    train_w2v[name]=df_embed_tr[name]
    test_w2v[name]=df_embed_te[name]

for name in w2v:
    train_w2v[name].to_csv("checkpoints_databases/new_stem_working_train_"+name+".csv",index=False)
    test_w2v[name].to_csv("checkpoints_databases/new_stem_working_test_"+name+".csv",index=False)

tfidf_w = TfidfVectorizer(
        min_df=3, max_features=8000, strip_accents=None, lowercase = False,
        analyzer='word', token_pattern=r'\w+', ngram_range=(1,3), use_idf=True,
        smooth_idf=True, sublinear_tf=True
        ).fit(train["Window_Text"])
tfidf_f = TfidfVectorizer(
        min_df=10, max_features=10000, strip_accents=None, lowercase = False,
        analyzer='word', token_pattern=r'\w+', ngram_range=(1,3), use_idf=True,
        smooth_idf=True, sublinear_tf=True
        ).fit(train["Full_Text"])

X_train_text_w = tfidf_w.transform(train["Window_Text"])
X_test_text_w = tfidf_w.transform(test["Window_Text"])
X_train_text_f = tfidf_f.transform(train["Full_Text"])
X_test_text_f = tfidf_f.transform(test["Full_Text"])

#tfidf_names =tfidf.get_feature_names()
#tfidf_names

tfidf_w.get_feature_names()


#same did thousands of time gridsearchs, perfect is 100 for our cases
dic_svd=TruncatedSVD(n_components=100,n_iter=25,random_state=26)

tsvd_train_w=dic_svd.fit_transform(X_train_text_w)
tsvd_test_w=dic_svd.transform(X_test_text_w)
tsvd_train_f=dic_svd.fit_transform(X_train_text_f)
tsvd_test_f=dic_svd.transform(X_test_text_f)
X_train_w=pd.DataFrame()
X_test_w=pd.DataFrame()
X_train_f=pd.DataFrame()
X_test_f=pd.DataFrame()
for i in range(int(100)):
    X_train_w['window_' +"tfidf_"+str(i)] = tsvd_train_w[:, i]
    X_test_w['window_' +"tfidf_"+str(i)] = tsvd_test_w[:, i]
    X_train_f['full_' +"tfidf_"+str(i)] = tsvd_train_f[:, i]
    X_test_f['full_' +"tfidf_"+str(i)] = tsvd_test_f[:, i]

X_train_wind=X_train_w
X_train_full=X_train_f
X_test_wind=X_test_w
X_test_full=X_test_f
dic_train={}
dic_test={}
dic_train["wind_tfidf_100"]=X_train_wind
dic_test["wind_tfidf_100"]=X_test_wind
dic_train["full_tfidf_100"]=X_train_full
dic_test["full_tfidf_100"]=X_test_full

for name in dic_train:
    dic_train[name].to_csv("checkpoints_databases/new_stem_working_train_"+name+".csv",index=False)
    dic_test[name].to_csv("checkpoints_databases/new_stem_working_test_"+name+".csv",index=False)

from gensim.models import KeyedVectors

w2v_bio = KeyedVectors.load_word2vec_format("../bases/PMC-w2v.bin",binary=True)

me_vec={}
me_train={}
me_test={}
me_vec = MeanEmbeddingVectorizer(w2v_bio)
me_train = me_vec.fit_transform(train['Full_Text'])
me_test = me_vec.fit_transform(test['Full_Text'])
df_bio_tr=pd.DataFrame(me_train)
df_bio_te=pd.DataFrame(me_test)

train_w2v_bio=df_bio_tr
test_w2v_bio=df_bio_te

train_w2v_bio.to_csv("checkpoints_databases/new_working_train_bio.csv",index=False)
test_w2v_bio.to_csv("checkpoints_databases/new_working_test_bio.csv",index=False)



