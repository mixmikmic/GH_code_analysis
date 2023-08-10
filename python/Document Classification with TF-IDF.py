get_ipython().magic('matplotlib inline')
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

post_texts = data.data
news_group_ids = data.target

print data.description

print "Number of posts", len(data.data)
import matplotlib.pyplot as plt
plt.hist(data.target, bins=20)
plt.xlabel('Newsgroup Id')
plt.ylabel('Number of Posts')
plt.show()

print "First post!"
print data.data[0]

import string

def tf(text):
    """ Returns a dictionary where keys are words that occur in text
        and the value is the number of times that each word occurs. """
    d = {}
    words = text.split()
    for w in words:
        modified_word = w.lower().strip('.,;!?"()')
        if not modified_word.strip(string.ascii_letters):
            d[modified_word] = d.get(modified_word,0) + 1
    return d

tf(data.data[0])

from math import log
import operator

def idf(data):
    """ Returns a dictionary where the keys are words and the values are inverse
        document frequencies.  For this function you should use the formula
        idf(w, data) = log(N / |text in data that contain the word w|) """
    document_count = {}
    for post in data:
        d = tf(post)
        for k in d:
            document_count[k] = document_count.get(k, 0) + 1
    
    idf = {}
    for key in document_count:
        idf[key] = log(len(data)/float(document_count[key]))
    return idf

idf = idf(data.data)
sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))

print "Lowest IDF (most common)"
for d in sorted_idf[0:10]:
    print d

print ""
print "Highest IDF (least common)"
rev_sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))
for d in reversed(rev_sorted_idf[-10:]):
    print d



