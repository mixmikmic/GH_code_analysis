import pandas as pd
import pickle

df_hadm_top10 = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')
ICD9CODES = pickle.load(open("./data/ICD9CODES.p", "r"))

df_hadm_top10.head(5)

len(df_hadm_top10)

import random

def separate(seed, N):    
    idx=list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train= idx[0:int(N*0.50)]
    idx_val= idx[int(N*0.50):int(N*0.75)]
    idx_test= idx[int(N*0.75):N]

    return idx_train, idx_val, idx_test


idx_train, idx_val, idx_test = separate(1234, df_hadm_top10.shape[0])
idx_join_train=idx_train + idx_val
len(idx_join_train)

df_hadm_top10_w2v=df_hadm_top10.iloc[idx_join_train].copy()
df_hadm_top10_w2v.head(5)

# Cleanning the data
# Light preprocesing done on purpose (so word2vec understand sentence structure)
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text.split()
    return text

token_review = list(df_hadm_top10_w2v['text'].apply(preprocessor))

len(token_review)

from gensim.models import Word2Vec
#import gensim.models.Word2Vec
from gensim import utils
from time import time

# assumptions: window is 5 words left and right, eliminate words than dont occur in
# more than 10 docs, use 4 workers for a quadcore machine. Size is the size of vector
# negative=5 implies negative sampling and makes doc2vec faster to train
# sg=0 means CBOW architecture used. sg=1 means skip-gram is used
#model = Doc2Vec(sentence, size=100, window=5, workers=4, min_count=5)


import random

size = 300  #change to 100 and 600 to generate vectors with those dimensions

#instantiate our  model
model_w2v = Word2Vec(min_count=10, window=5, size=size, sample=1e-3, negative=5, workers=4, sg=0)

#build vocab over all reviews
model_w2v.build_vocab(token_review)

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
Idx=list(range(len(token_review)))

t0 = time()
for epoch in range(5):
     random.shuffle(Idx)
     perm_sentences = [token_review[i] for i in Idx]
     model_w2v.train(perm_sentences)
     print(epoch)
    
elapsed=time() - t0
print("Time taken for Word2vec training: ", elapsed, "seconds.")

# saves the word2vec model to be used later.
#model_w2v.save('./model_word2vec_skipgram_300dim')

# open a saved word2vec model 
#import gensim
#model_w2v=gensim.models.Word2Vec.load('./model_word2vec')


#model_w2v.wv.save_word2vec_format('./model_word2vec_v2_300dim.txt', binary=False)


model_w2v.wv.most_similar('cancer')

# Run this cell if you are using Glove type format
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

model_w2v=loadGloveModel("./data/model_word2vec_v2_300dim.txt")

import pickle
import pandas as pd

df_hadm_top10 = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')
ICD9CODES = pickle.load(open("./data/ICD9CODES.p", "r"))
df_hadm_top10.head(5)

# Cleanning the data
import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    text = re.sub(" \d+", " ", text)
    return text


df_hadm_top10['text2'] = df_hadm_top10['text'].apply(preprocessor)

# Create tokens
token_review=[]
for i in range(df_hadm_top10['text2'].shape[0]):
    review = df_hadm_top10['text2'][i]
    token_review.append([i for i in review.split()])

len(token_review)

import numpy as np  # Make sure that numpy is imported
from nltk.corpus import stopwords

STOPWORDS_WORD2VEC = stopwords.words('english') + ICD9CODES

keys_updated = [word for word in model_w2v.keys() if word not in STOPWORDS_WORD2VEC]
index2word_set=set(keys_updated)

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    #index2word_set = set(model.wv.index2word) #activate if using gensim

    # activate if uploaded text version
    #index2word_set=set(keys_updated)
    
    
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

#token_review[200]

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 10000th review
       if counter%10000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

from time import time
t0 = time()
final_w2v=getAvgFeatureVecs(token_review, model_w2v, num_features=300)
elapsed=time() - t0
print("Time taken for Word2vec avg vector per note calculation: ", elapsed, "seconds.")

len(final_w2v)

labels=["id", `4019`, `2724`,`25000`,`4280`,`41401`,`53081`,`51881`,`42731`,`5849`,`5990`]
len(df_hadm_top10[labels])

# Create train set and test set to use Machine Learning model
import random

final_w2v_df=pd.DataFrame(data=final_w2v)  
data_final=pd.concat([df_hadm_top10[labels],final_w2v_df], axis=1)

idx=list(range(len(token_review)))
random.seed(1234)
random.shuffle(idx)
idx_train= idx[0:int(len(data_final)*0.50)]
idx_val= idx[int(len(data_final)*0.50):int(len(data_final)*0.75)]
idx_test= idx[int(len(data_final)*0.75):len(data_final)]

train_set = data_final.iloc[idx_train]
val_set = data_final.iloc[idx_val]
test_set =data_final.iloc[idx_test]

print(train_set.shape, val_set.shape  ,test_set.shape)



