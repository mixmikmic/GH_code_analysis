get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('precision 4')
import os, sys, glob
import regex as re
import string

import requests
url = "http://www.gutenberg.org/cache/epub/35534/pg35534.txt"
raw = requests.get(url).text

# peek at the first 1000 characters of the downloaded text
raw[:1000]

start = re.search(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*", raw).end()
stop = re.search(r"End of the Project Gutenberg EBook", raw).start()
text = raw[start:stop]
text[:1000]

# A naive but workable approach would be to first strip all punctuation, 
# convert to lower case, then split on white space
words1 = re.sub(ur"\p{P}+", "", text.lower()).split()
print words1[:100]
len(words1)

# If you need to be more careful, use the nltk tokenizer.
import nltk
from multiprocessing import Pool
from itertools import chain
punkt = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = punkt.tokenize(text.lower())
# since the tokenizer works on a per sentence level, we can parallelize
p = Pool()
words2 = list(chain.from_iterable(p.map(nltk.tokenize.word_tokenize, sentences)))
p.close()
# Now remove words that consist of only punctuation characters
words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]
# Remove contractions - wods that begin with '
words2 = [word for word in words2 if not (word.startswith("'") and len(word) <=2)]
print words2[:100]
len(words2)

from collections import Counter
c = Counter(words2)
c.most_common(n=10)

# this isn't very helpful since there are many "stop" words that don't man much
# now just the top 10 wordss give a good idea of what the book is about!
stopwords = nltk.corpus.stopwords.words('english')
new_c = c.copy()
for key in c:
    if key in stopwords:
        del new_c[key]
new_c.most_common(n=10)

# words in words1 but not in words2
w12 = list(set(words1) - set(words2))
w12[:10]

# words in word2 but not in word1
w21 = list(set(words2) - set(words1))
w21[:10]

get_ipython().magic('load_ext version_information')

get_ipython().magic('version_information requests, regex, nltk')



