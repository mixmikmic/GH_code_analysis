import string

from nltk.stem.wordnet import WordNetLemmatizer

from gensim import models
from gensim.corpora import Dictionary

## Example text:
zen = ["Beautiful is better than ugly. Explicit is better than implicit.", 
        "Simple is better than complex. Complex is better than complicated.",
        "Flat is better than nested. Sparse is better than dense.",
        "Readability counts. Special cases aren't special enough to break the rules.",
        "Although practicality beats purity. Errors should never pass silently.",
        "Unless explicitly silenced. In the face of ambiguity, refuse the temptation to guess." ,
        "There should be one-- and preferably only one --obvious way to do it.",
        "Although that way may not be obvious at first unless you're Dutch.",
        "Now is better than never. Although never is often better than *right* now.",
        "If the implementation is hard to explain, it's a bad idea.",
        "If the implementation is easy to explain, it may be a good idea."
        "Namespaces are one honking great idea -- let's do more of those!"]

## TOKENISATION: Breaking down a text in meaningful elements
texts = [text.lower().replace('\n', ' ').split(' ') for text in zen]

stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for'
             'from', 'has', 'he', 'if', 'in', 'is', 'it', 'its', 'it\'s', 'of', 'on', 
             'than', 'that', 'the', 'to', 'was', 'were', 'will', 'with']

docs = [[filter(lambda x:x not in string.punctuation, i) for i in txt if i != '' and i not in stop_words] 
        for txt in texts]
print docs

## LEMMATISATION:
## Grouping together the different forms of a word
lmtzr = WordNetLemmatizer()
lemm = [[lmtzr.lemmatize(word) for word in data] for data in docs]
print lemm

## Create bag of words from dictionnary:
####note: compare doc2bow and word2vec
dictionary = Dictionary(lemm)
dictionary.save('../dicts/test-text.dict')

## Term frequency–inverse document frequency (TF-IDF)
## Method to reflect how important a word is to a document in a collection.
## The inverse document frequency measures whether the term is common or rare across all documents.

bow = [dictionary.doc2bow(l) for l in lemm] # Calculates inverse document counts for all terms
print "BAG OF WORDS: Assign a frequency to a word index \n", bow

# Transform the count representation into the Tfidf space
tfidf = models.TfidfModel(bow)              
corpus_tfidf = tfidf[bow]
print "\nTF-IDF: value associated with the importance of word in a document\n"
for doc in corpus_tfidf:
    print doc

## Build the LSI (latent semantic indexing) model
## Method to uses a mathematical technique called singular value decomposition (SVD) 
## to identify patterns in the relationships between the terms and concepts contained 
## in an unstructured collection of text.
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
corpus_lsi = lsi[corpus_tfidf]

for doc in corpus_lsi:
    print(doc)

for i in range(lsi.num_topics):
    print lsi.show_topic(i)

list_topics = [] 
for i in range(lsi.num_topics):
    list_topics.extend(lsi.show_topic(i))

list_topics.sort(key=lambda tup: tup[0], reverse=True)

topics = [i[1] for i in list_topics]
print topics



