import pandas as pd
import numpy as np
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

conn = MongoClient()
db = conn.project

db.recipes.count()

cursor = db.recipes.find({}, {'title': 1, 'ingredients': 1, '_id' : 0})
for doc in cursor:
    print doc
#     print doc['title']
#     print "; ".join(doc['ingredients'])

cursor = db.recipes.find({}, {'title': 1, 'ingredients': 1, '_id' : 0})
data = pd.DataFrame(list(cursor))

data.head()

data['ingredients'] = data['ingredients'].apply(lambda x: " ".join(x))

data['ingredients']

documents = data['ingredients'].values

documents[:2]

documents[0]

list_ = [[word for word in document.lower().split() if not any(c.isdigit() for c in word)] for document in documents]
print list_[0]

any(c.isdigit() for c in '30ml50ml')

import string
string.punctuation

wnl = WordNetLemmatizer()
stopset = set(stopwords.words('english'))

list_lem = [[wnl.lemmatize(word.strip(string.punctuation)) for word in document.lower().replace("/", "").split()             if word not in stopset if not word.isdigit()] for document in documents]
print list_lem[0]

def clean_text(documents):
    '''
    INPUT: array of strings
    OUTPUT: array of lists (ok?)
    '''
    stopset = set(stopwords.words('english'))
    stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
    wnl = WordNetLemmatizer()
    texts = [[wnl.lemmatize(word.strip(string.punctuation)) for word in document.lower().replace("/", "").split()             if word not in stopset if not word.isdigit()] for document in documents]
    text_array = np.array(texts)
    return text_array

text_array = clean_text(documents)

text_array[0]









dictionary = corpora.Dictionary(texts)

count = 0
for i in dictionary.iteritems():
    count +=1

print count

print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]

corpus[:2]

tfidf = models.TfidfModel(corpus)
index = similarities.Similarity(tfidf[corpus]) ## prepare for similarity queries

cursor_test = db.restaurants.find({}, {'name':1, '_id' :0}).limit(10)

for i in cursor_test:
    print i

r_cursor = db.restaurants.find_one({'name' : 'Nopa'})

menu = r_cursor['menu']

menu_list = []
for i in zip(menu['items'], menu['descriptions']):
    menu_list.append(': '.join(i))

menu_string = ' '.join(menu_list)

menu_string

menu_tokens = [word for word in menu_string.lower().split() if word not in stoplist]  #add description,  available to stopwords?
menu_vector = dictionary.doc2bow(menu_tokens)

sims = index[tfidf[menu_vector]]  ## kills kernel :(  Try running from terminal?
## will argmax to get position and refer back to recipes df to get title.

type(stoplist)

stoplist.add('description')

stoplist.update(['description', 'available'])

stoplist

## Load the model
#model = models.Word2Vec.load_word2vec_format('models/glove.6B.50d.txt', binary=False)
# fn = "models/GoogleNews-vectors-negative300.bin.gz" # pretrained on Google News
# model = models.Word2Vec.load_word2vec_format(fn, binary= True) ## takes too long..

## Convert words to vectors
## For restaurants texts
# menu_vector = model[word] # not by word but by doc?
# ## For recipe texts
# index = similarities.Similarity(model[corpus]) ## corpus or texts?

## Find most similar 
# sims = index[model[menu_vector]] ## convert BOW to Tfidf
# rec_indices = np.argsort(sims)[:-num:-1] # gets top 10 ## do I want to report similarity score too?
# data.loc[rec_indices, 'title'], sims[rec_indices]

# print model.most_similar('vacation')



