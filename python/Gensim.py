from gensim import corpora

# This is a tiny corpus of nine documents, each consisting of only a single sentence.
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",              
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# Remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

print(texts)

# Remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
        
texts = [[token for token in text if frequency[token] > 1] for text in texts]

print(texts)

#pretty-printer
from pprint import pprint
pprint(texts)

dictionary = corpora.Dictionary(texts)
# we assign a unique integer ID to all words appearing in the processed corpus
# this sweeps across the texts, collecting word counts and relevant statistics.
# In the end, we see there are twelve distinct words in the processed corpus, which means each document will be represented by twelve numbers (ie., by a 12-D vector).


print(dictionary)

# To see the mapping between the words and their ids
print(dictionary.token2id)

# To convert tokenized documents to vectors
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())

print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
for c in corpus:
    print(c)

