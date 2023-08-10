import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
pd.__version__
import sys
from scipy import stats
import time
import  pysparse
from scipy.linalg import norm
import sompylib.sompy as SOM

get_ipython().magic('matplotlib inline')

with open('./Data/IMDB_data/pos.txt','r') as infile:
    reviews = infile.readlines()
len(reviews)

reviews[0]

def cleanText(corpus):
    import string
    validchars = string.ascii_letters + string.digits + ' '
    punctuation = """.,:;?!@(){}[]$1234567890"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    
    for c in punctuation:
        corpus =[z.replace(c, '') for z in corpus]
    

    corpus = [''.join(ch for ch in z if ch in validchars) for z in corpus]
    
    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
#     corpus = [z.split() for z in corpus]
    corpus = [z.replace(' ', '_') for z in corpus]
    return corpus

texts  = cleanText(reviews)

texts[0]

all_chars = '_abcdefghijklmnopqrstuvwxyz'
dictionary = {}
for i in range(len(all_chars)):
    dictionary[all_chars[i]] = i
dictionary

# building data with the format of sequence
data = []
for text in texts[:]:
    d = []
    for c in text:
        d.append(dictionary[c])
    data.append(d)
print len(data)

def buildTM_from_sequential_data(data,states,irreducible=True):
    # each row is a sequence of observation
    n = len(states)
    M = np.zeros((n,n))
    for d in data:
        for k in range(1,len(d)):
            i = d[k-1]
            
            j = d[k]
            M[i,j]= M[i,j] + 1
    
    eps = .001
    for i in range(M.shape[0]):
        s= sum(M[i])
        
        if s==0:
            if irreducible==True:
                M[i]=eps
                M[i,i]=1.
                s= sum(M[i])
                M[i]=np.divide(M[i],s)
            else:
                M[i,i]=1.
        else:
            M[i]=np.divide(M[i],s)    
    return M


# Power iteration Method
def simulate_markov(TM,verbose='on'):
    e1 = time.time()
    states_n = TM.shape[0]
    pi = np.ones(states_n);  pi1 = np.zeros(states_n);
    pi = np.random.rand(states_n)
   
    pi = pi/pi.sum()
    n = norm(pi - pi1); i = 0;
    diff = []
    while n > 1e-6 and i <1*1e4 :
        pi1 = TM.T.dot(pi).copy()
        n = norm(pi - pi1); i += 1
        diff.append(n)
        pi = pi1.copy()
    if verbose=='on':
        print "Iterating {} times in {}".format(i, time.time() - e1)
    
    mixing_ = i
    return pi1,mixing_

states = np.unique(dictionary.values())
M_char = buildTM_from_sequential_data(data,states,irreducible=True)

chars = np.asarray([c for c in all_chars])

pi,mixing_ = simulate_markov(M_char,verbose='on')
plt.plot(pi);
plt.xticks(range(27),chars);
plt.grid()

# To see if we can generate something
n_state = M_char.shape[0]
ind_initial = np.random.randint(0,n_state,size=1)
print chars[ind_initial[0]]
ind = ind_initial[0]
for i in range(20):

    
    # If we take the most likely next chars, it quickly falls in a loop?!!
    ind = np.argmax(M_char[ind])
    
    
    # If we take the next char based on a random choice based on the probabilites 
#     ind = np.random.choice(range(M_char.shape[0]),size=1,p=M_char[ind])[0]
    
    print chars[ind]

# To see if we can generate something
n_state = M_char.shape[0]
ind_initial = np.random.randint(0,n_state,size=1)
print chars[ind_initial[0]]
ind = ind_initial[0]
for i in range(20):

    
#     If we take the most likely next chars, it quickly falls in a loop?!!
#     ind = np.argmax(M_char[ind])
    
    
    # If we take the next char based on a random choice based on the probabilites 
    ind = np.random.choice(range(M_char.shape[0]),size=1,p=M_char[ind])[0]
    
    print chars[ind]

# codes from https://github.com/codebox/markov-text
import sys

sys.path.insert(0, './markovtext')

from db import Db
from gen import Generator
from parse import Parser
from sql import Sql
from rnd import Rnd
import sys
import sqlite3
import codecs





SENTENCE_SEPARATOR = '.'
WORD_SEPARATOR = ' '

args = ['','gen','IMDB2','2']

if (len(args) < 3):
	raise ValueError(usage)
mode  = 'gen'
name  = './markovtext/IMDB_N2'
count = 4


if mode == 'parse':
    
    depth = 2
    file_name = './Data/IMDB_data/pos.txt'

    db = Db(sqlite3.connect(name + '.db'), Sql())
    db.setup(depth)

    txt = codecs.open(file_name, 'r', 'utf-8').read()
    Parser(name, db, SENTENCE_SEPARATOR, WORD_SEPARATOR).parse(txt)

elif mode == 'gen':    
    db = Db(sqlite3.connect(name + '.db'), Sql())
    generator = Generator(name, db, Rnd())
    for i in range(0, count):
        print "{}\n".format(i)
        print generator.generate(WORD_SEPARATOR)
        

else:
	raise ValueError(usage)

# For each char
ind_initial = np.random.randint(0,n_state,size=1)[0]

print 'the selected char: {}'.format(chars[ind_initial])
plt.plot(range(M_char.shape[0]),M_char[ind_initial],'.-');

plt.xticks(range(M_char.shape[0]),chars);
plt.grid();

import sompylib.sompy as SOM

msz11 =20
msz10 = 20

X = M_char

som_char = SOM.SOM('', X, mapsize = [msz10, msz11],norm_method = 'var',initmethod='pca')
# som1 = SOM1.SOM('', X, mapsize = [msz10, msz11],norm_method = 'var',initmethod='pca')
som_char.init_map()
som_char.train(n_job = 1, shared_memory = 'no',verbose='final')
codebook_char = som_char.codebook[:]
codebook_char_n = SOM.denormalize_by(som_char.data_raw, codebook_char, n_method = 'var')

# we projects all the vectors in SOM and visualize it 
xy = som_char.ind_to_xy(som_char.project_data(X))
xy

ax = plt.subplot(1,1,1)
for i in range(len(X)):
    plt.annotate(chars[i], (xy[i,1],xy[i,0]),size=20, va="center")
    plt.xlim((0,som_char.mapsize[0]))
    plt.ylim((0,som_char.mapsize[0]))
plt.xticks([])
plt.yticks([])

## Just a hint to Gradient Descent

x_old = 0
x_new = 9
eps = .001
precision = .00001

# y = x^2
def f(x):
    return np.power(x,4) -3*np.power(x,3)  + 2

def f_deriv(x):
    return 4*np.power(x,3) - 9*x



counter = 0
while abs(x_old-x_new)>precision:
    x_old = x_new
    x_new = x_old - eps*f_deriv(x_old)
    plt.plot(x_new,f(x_new),'or')
    counter = counter + 1
print x_new 

for x in np.linspace(-10,10,100):
    plt.plot(x,f(x),'.b',markersize=1)

#### We use a beautiful library called, gensim
import gensim
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# from gensim.models import word2vec
# # get the pretrained vector from https://code.google.com/archive/p/word2vec/
# Google_w2v = word2vec.Word2Vec.load_word2vec_format('/Users/SVM/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

# print Google_w2v.most_similar(['girl', 'father'], ['boy'], topn=1)
# print Google_w2v.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)

get_ipython().magic('matplotlib inline')
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
with open('/All_Files/Files/Data/gensim/sample_Data/IMDB_data/pos.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open('/All_Files/Files/Data/gensim/sample_Data/IMDB_data/neg.txt', 'r') as infile:
    neg_tweets = infile.readlines()
    
with open('/All_Files/Files/Data/gensim/sample_Data/IMDB_data/unsup.txt','r') as infile:
    unsup_reviews = infile.readlines()

#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.5)

#Do some very minor text preprocessing



def cleanText(corpus):
    import string
    validchars = string.ascii_letters + string.digits + ' '
    punctuation = """.,:;@(){}[]$1234567890"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    
    for c in punctuation:
        corpus =[z.replace(c, '') for z in corpus]
    

    corpus = [''.join(ch for ch in z if ch in validchars) for z in corpus]
    
    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
#     corpus = [z.replace(' ', '_') for z in corpus]
    return corpus


x_train_c = cleanText(x_train)
x_test_c = cleanText(x_test)
unsup_  = cleanText(unsup_reviews)


n_dim = 150
#Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10,
                    sentences=None, alpha=0.025, window=5, max_vocab_size=None,
                    sample=0, seed=1, workers=6, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0,
                    iter=1, null_word=0)


# imdb_w2v.build_vocab(np.concatenate((unsup_,x_train)))
imdb_w2v.build_vocab(x_train_c)

# Train the model over train_reviews (this may take several minutes)
# imdb_w2v.train(np.concatenate((unsup_,x_train)))
imdb_w2v.train(x_train_c)

print imdb_w2v.most_similar(['good'], topn=5)
# print imdb_w2v.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)

import pandas as pd
import pandas.io.data
import numpy as np
from matplotlib import pyplot as plt
import sys
from sklearn.preprocessing import scale

pd.__version__

# Here, we have the choice of using ay pretrained model too
Googlevec = 'No'
import gensim


vocablen = len(imdb_w2v.vocab.keys())

# vocablen = len(uniq_from_x_train)

vector_size = imdb_w2v.vector_size
VocabVec = np.zeros((vocablen,vector_size))

vocab = imdb_w2v.vocab.keys()



for i in range(vocablen):
    if Googlevec=='Yes':
        try:
            VocabVec[i] = Google_w2v[vocab[i]]
        except:
            continue
    else:
        try:
            VocabVec[i] = imdb_w2v[vocab[i]]
        except:
            continue
        
print Googlevec 

print 'data size', VocabVec.shape

def buildDocHistogram(Vocab_ind, text, ind_size,normalize='Yes'):
    vec = np.zeros(ind_size).reshape((1, ind_size))
    count = 0.
    for word in text:
        try:
            vec[0,Vocab_ind[word]] += 1
            count += 1.
        except KeyError:
            continue
    if count != 0:
        if normalize=='Yes':
            vec /= count
    return vec

#Build new dim for vocabs based on SOMinds
print 'data size', VocabVec.shape
ind_final_vocab = VocabVec.sum(axis=1)!=0

final_VocabVec = VocabVec[ind_final_vocab]

final_vocab = list(np.asarray(vocab)[ind_final_vocab])
Vocab_Wordind = dict(zip(final_vocab,range(len(final_vocab)) ))

ind_size = len(final_vocab)
labels = y_train
ind_pos = labels==1
ind_neg = labels==0
all_coocur_ = np.zeros((len(x_train_c),ind_size))
for i in range(len(x_train_c)):
    all_coocur_[i]= buildDocHistogram(Vocab_Wordind, x_train_c[i], ind_size,normalize='No')
all_coocur_  = all_coocur_.sum(axis=0)

print 'all done'
len_neg = len(list(np.asarray(x_train_c)[ind_neg]))
neg_coocur_ = np.zeros((len_neg,ind_size))
for i,text in enumerate(list(np.asarray(x_train_c)[ind_neg])):
    neg_coocur_[i,:]= buildDocHistogram(Vocab_Wordind, text, ind_size,normalize='No')
neg_coocur_  = neg_coocur_.sum(axis=0)

print 'neg done'
len_pos = len(list(np.asarray(x_train_c)[ind_pos]))
pos_coocur_ = np.zeros((len_pos,ind_size))
for i,text in enumerate(list(np.asarray(x_train_c)[ind_pos])):
    pos_coocur_[i,:]= buildDocHistogram(Vocab_Wordind, text, ind_size,normalize='No')
pos_coocur_  = pos_coocur_.sum(axis=0)

print 'pos done'

labels = y_train
ind_pos = labels==1
ind_neg = labels==0
#Make the histogram of documents basedo n SOMinds
ind_size = len(final_vocab)


# all_coocur_ = np.concatenate([buildDocHistogram(Vocab_Wordind, z, ind_size,normalize='No') for z in x_train_c])
# pos_coocur_ = np.concatenate([buildDocHistogram(Vocab_Wordind, z, ind_size,normalize='No') for z in list(np.asarray(x_train_c)[ind_pos])])
# neg_coocur_ = np.concatenate([buildDocHistogram(Vocab_Wordind, z, ind_size,normalize='No') for z in list(np.asarray(x_train_c)[ind_neg])])


# #Summing over all texts for each word
# pos_coocur_ = pos_coocur_.sum(axis=0)
# neg_coocur_ = neg_coocur_.sum(axis=0)
# all_coocur_ = all_coocur_.sum(axis=0)

#normalizing the values
# pos_coocur_ = pos_coocur_/all_coocur_
# neg_coocur_ = neg_coocur_/all_coocur_

pos_to_neg = pos_coocur_/(neg_coocur_+1)

sorted_features =pd.DataFrame(index=range(pos_coocur_.shape[0]))
sorted_features['words'] = Vocab_Wordind.keys()
sorted_features['pos_coocur_'] = pos_coocur_
sorted_features['neg_coocur_'] = neg_coocur_
sorted_features['pos_to_neg'] = pos_to_neg
sorted_features['differ'] = np.abs(neg_coocur_-pos_coocur_)
sorted_features = sorted_features.sort_values('differ',ascending=False)
sorted_features.head()

sorted_features.shape


###############
###############


### It seems that having all the features is not that bad! Even the results are similar, eventhough it might slow down the 
### som trainig and som projection steps, it dosne't need conditional probabilities to be calculated
sel_features = sorted_features.index[:15000].values
Data= final_VocabVec[sel_features,:]


# sel_features = sorted_features.index[:].values
# Data= final_VocabVec

# len(sel_vocab)


#Train a SOM based on vocabs
# reload(sys.modules['sompy'])
ind_size = 3000
sm1 = SOM.SOM('sm', Data, mapsize = [1,ind_size],norm_method = 'var',initmethod='pca')
# ind_size = 50*50
sm1.train(n_job = 1, shared_memory = 'no',verbose='final')
print 'Training Done'

# sm1.hit_map()
print sm1.codebook.shape

#Remained Data
print sm1.data.shape


#Build new dim for vocabs based on SOMinds
Vocab_Somind = dict(zip(list(np.asarray(final_vocab)[sel_features]), list(sm1.project_data(Data))))
# Vocab_Somind = dict(zip(list(np.asarray(final_vocab)[:]), list(sm1.project_data(Data))))


# Vocab_Somind = dict(zip(final_vocab, list(sm1.project_data(Data))))

DF = pd.DataFrame()
DF['word']=np.asarray(final_vocab)[sel_features]
b = sm1.project_data(Data)
DF['somind'] = b

DF.sort_values('somind')[:10]


from sklearn.preprocessing import scale
#Make the histogram of documents basedo n SOMinds
train_vecs = np.concatenate([buildDocHistogram(Vocab_Somind, z, ind_size) for z in x_train_c])
train_vecs = scale(train_vecs)


test_vecs = np.concatenate([buildDocHistogram(Vocab_Somind, z, ind_size) for z in x_test_c])
test_vecs = scale(test_vecs)


# #now select the most informative features (here are sominds, but we can do this on original words too)

# def calc_conditional_feature_importance(corpus_mat,labels):
#     #corpus_mat is the original matrix where each row is one record and columns are features, where are either words or sominds
#     #sentiments are labels
#     #it returns a matrix showing the relative importance of each feature regarding to each label
#     pos_coocur_ = np.zeros((corpus_mat.shape[1],1))
#     neg_coocur_ = np.zeros((corpus_mat.shape[1],1))
#     ind_pos = labels==1
#     ind_neg = labels==0
#     for i in range(corpus_mat.shape[1]):
#         pos_coocur_[i] = np.sum(corpus_mat[ind_pos,i])
#         neg_coocur_[i] = np.sum(corpus_mat[ind_neg,i])
#         sum_ = (pos_coocur_[i]+neg_coocur_[i])
#         if sum_ !=0:
#             pos_coocur_[i] = pos_coocur_[i]/sum_
#             neg_coocur_[i] = neg_coocur_[i]/sum_
        
# #             print i
#     DF =pd.DataFrame(index=range(corpus_mat.shape[1]))
#     DF['pos_coocur_'] = pos_coocur_
#     DF['neg_coocur_'] = neg_coocur_
#     DF['differ'] = np.abs(neg_coocur_-pos_coocur_)
#     DF = DF.sort_values('differ',ascending=False)
#     return DF

#Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
from sklearn.linear_model import SGDClassifier
import sklearn.linear_model as lm
lm.RidgeClassifier
from sklearn.decomposition import RandomizedPCA


# howmany = range(10,sm1.nnodes,200)
# # howmany = range(10,15000,500)
# howmany = range(sm1.nnodes,sm1.nnodes+1)
clf = lm.RidgeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = lm.SGDClassifier(loss="hinge", alpha=0.01, n_iter=200)

# import sklearn.ensemble as ensemble
# clf = ensemble.RandomForestRegressor(n_jobs=1) 






X_Train = train_vecs[:]
X_Test = test_vecs[:]

# pca = RandomizedPCA(n_components=int(.05*X_Train.shape[1]))
# pca.fit(X_Train)
# X_Train = pca.transform(X_Train)
# X_Test = pca.transform(X_Test)




clf.fit(X_Train, y_train)




import sklearn.metrics as metrics
print metrics.classification_report(y_test,clf.predict(X_Test))

















