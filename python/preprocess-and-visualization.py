import matplotlib.pyplot as plt
import nltk
import numpy as np

from nltk import FreqDist
from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')

text = "Hello I am paradox. I exist; therefore i may not. Existence is merely a thought process, right?"

# simple tokenization
words = text.split()
print(words)
frequency_map = Counter(words)

# for scatter plot
indices_max = 20 # max number of points to plot
Y = list(frequency_map.values())[:indices_max]
X = list(range(len(Y)))
words_plot = list(frequency_map.keys())[:indices_max]

plt.scatter(X, Y)

# scatter plot with labelled point
fig, ax = plt.subplots()
ax.scatter(X, Y)

for i, txt in enumerate(words_plot):
    ax.annotate(txt, (X[i],Y[i]))

# nltk tokenizer and frequency map
text = "Hello I am   paradox. I exist; therefore i may not. Existence is merely a thought process, right?\ni am"
tokens = nltk.word_tokenize(text)
print(tokens)
freq = FreqDist(tokens)
print(freq['i'])
freq.plot(20, cumulative=False)

# custom sentence tokenization
import re
sentences = re.split(r'[.?/\n]+', text)
print(sentences)

# nltk sentence level tokenization
sentences = nltk.sent_tokenize(text)
print(sentences)

# nltk pos tagging
text = "My name is Paradox"
tokens = nltk.word_tokenize(text)
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)

nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('NN')
nltk.help.upenn_tagset('VBZ')
nltk.help.upenn_tagset('NNP')

from nltk.stem.porter import PorterStemmer

# PorterStemmer is based on Porter Stemming Algorithm
porter_stemmer = PorterStemmer()
text = "I am Paradox. Most of the time i just sit quietly here and there - contemplating and thinking about existence. i tend to have many cups of caffeine."
tokens = nltk.word_tokenize(text)
print(tokens)
stemmed = [ porter_stemmer.stem(token) for token in tokens]
print(stemmed)

from nltk.stem.lancaster import LancasterStemmer

# it is based on Lancaster Stemming Algorithm
lancaster_stemmer = LancasterStemmer()
stemmed_lanc = list(map(lancaster_stemmer.stem, tokens))
print(stemmed_lanc)

from nltk.stem import WordNetLemmatizer

text = "I am Paradox. Most of the time i just sit quietly here and there - contemplating and thinking about existence. i tend to have many cups of caffeine."
tokens = nltk.word_tokenize(text)
wordnet_lemmatizer = WordNetLemmatizer()
lemmas = [ wordnet_lemmatizer.lemmatize(token, pos='v') for token in tokens ]
print(lemmas)

words = ['operation', 'operating', 'operational', 'abandonment', 'is', 'are']
stems = list(map(porter_stemmer.stem, words))
lemmas = [ wordnet_lemmatizer.lemmatize(word, pos='v') for word in words ]
print(stems)
print(lemmas)

