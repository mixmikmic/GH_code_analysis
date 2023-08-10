from sklearn.datasets import fetch_20newsgroups
rdd = sc.parallelize(fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes')).data)

n_docs = rdd.count()
print n_docs

#import nltk -> if you need to download the stopword corpa, here's how to call the interface
#nltk.download()

def unique_doc_words(doc, stop_words):
    l_words = filter(lambda x: x not in stop_words.value and x != '',
                        map(lambda word: word.strip(".,-;?").lower(), doc.split()))
    return list(set(l_words))

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
print list(stop)[:5]

bc_stop = sc.broadcast(stop)

rdd_term_docs = rdd.flatMap(lambda doc: unique_doc_words(doc, bc_stop)).cache()
rdd_term_docs.count()

from math import log10

#Remember to create broadcast variables!
bc_n_docs = sc.broadcast(n_docs)

def idf(doc_freq, n_docs):
    return log10((1+n_docs.value)/(1+doc_freq))

df = rdd_term_docs.map(lambda word: (word, 1))            .reduceByKey(lambda a, b: a+b)            .map(lambda x: (x[0], idf(x[1], bc_n_docs)))            .collectAsMap()
            
print list(df.iteritems())[:5]

