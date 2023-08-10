import nltk

nltk.__version__

moby = nltk.text.Text(nltk.corpus.gutenberg.words('melville-moby_dick.txt'))
moby

# The corcordance function performs a search for the given token and then also provides the surrounding context
moby.concordance('monstrous', 55, lines=10)

print (moby.similar("ahab"))
austen = nltk.text.Text(nltk.corpus.gutenberg.words('austen-sense.txt'))
print ()
print (austen.similar("monstrous"))

moby.common_contexts(['ahab','starbuck'])

inaugural = nltk.text.Text(nltk.corpus.inaugural.words())
inaugural.dispersion_plot(['citizens','democracy','freedom','duties','America'])

# Lists the varisou corpora and CorpusReader classes in the nltk.corpus module
for name in dir(nltk.corpus):
    if name.islower() and not name.startswith('_'): print (name)

# For a specific corpus and CorpurReader classes in the nltk.corpus module
nltk.corpus.shakespeare.fileids()

print (nltk.corpus.gutenberg.fileids())

print (nltk.corpus.stopwords.fileids())

nltk.corpus.stopwords.fileids()
import string
print (string.punctuation)

corpus = nltk.corpus.brown
print (corpus.paras())

print (corpus.sents())

print (corpus.words())

print (corpus.words())

print (corpus.raw()[:200]) # Be Careful!

reuters = nltk.corpus.reuters # Corpus of news articles
counts = nltk.FreqDist(reuters.words())
vocab = len(counts.keys())
words = sum(counts.values())
lexdiv = float(words) / float(vocab)

print ("Corpus has %i types and %i tokens for a lexical diversity of %0.3f" % (vocab, words, lexdiv))

counts.B()

print(counts.most_common(40)) # The n most common tokens in the corpus

print(counts.max) # The most frequent token in the corpus

print(counts.hapaxes()[0:10]) # A list of all the hapax legomena

counts.freq('stipulate') * 100 # Precentage of the corpus for this token

counts.plot(200, cumulative=False)

from itertools import chain 

brown = nltk.corpus.brown
categories = brown.categories()

counts = nltk.ConditionalFreqDist(chain(*[[(cat, word) for word in brown.words(categories=cat)] for cat in categories]))

for category, dist in counts.items():
    vocab  = len(dist.keys())
    tokens = sum(dist.values())
    lexdiv = float(tokens) / float(vocab)
    print ("%s: %i types with %i tokens and lexical diveristy of %0.3f" % (category, vocab, tokens, lexdiv))

for ngram in nltk.ngrams(["The", "bear", "walked", "in", "the", "woods", "at", "midnight"], 5):
    print (ngram)

text = u"Medical personnel returning to New York and New Jersey from the Ebola-riddled countries in West Africa will be automatically quarantined if they had direct contact with an infected person, officials announced Friday. New York Gov. Andrew Cuomo (D) and New Jersey Gov. Chris Christie (R) announced the decision at a joint news conference Friday at 7 World Trade Center. “We have to do more,” Cuomo said. “It’s too serious of a situation to leave it to the honor system of compliance.” They said that public-health officials at John F. Kennedy and Newark Liberty international airports, where enhanced screening for Ebola is taking place, would make the determination on who would be quarantined. Anyone who had direct contact with an Ebola patient in Liberia, Sierra Leone or Guinea will be quarantined. In addition, anyone who traveled there but had no such contact would be actively monitored and possibly quarantined, authorities said. This news came a day after a doctor who had treated Ebola patients in Guinea was diagnosed in Manhattan, becoming the fourth person diagnosed with the virus in the United States and the first outside of Dallas. And the decision came not long after a health-care worker who had treated Ebola patients arrived at Newark, one of five airports where people traveling from West Africa to the United States are encountering the stricter screening rules."

for sent in nltk.sent_tokenize(text):
    print(sent)
    print()

for sent in nltk.sent_tokenize(text):
    print(list(nltk.wordpunct_tokenize(sent)))
    print()

for sent in nltk.sent_tokenize(text):
    print(list(nltk.pos_tag(nltk.word_tokenize(sent))))

for sent in nltk.sent_tokenize(text):
    print(list(nltk.pos_tag(nltk.word_tokenize(sent))))
    print()

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer

text = list(nltk.word_tokenize("The women running in the fog passed bunnies working as computer scientists."))

snowball = SnowballStemmer('english')
lancaster = LancasterStemmer()
porter = PorterStemmer()

for stemmer in (snowball, lancaster, porter):
    stemmed_text = [stemmer.stem(t) for t in text]
    print(" ".join(stemmed_text))

from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(t) for t in text]
print (" ".join(lemmas))

import string

# Module constants
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
punctuation = string.punctuation

def normalize(text):
    for token in nltk.word_tokenize(text):
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        if token not in stopwords and token not in punctuation:
            yield token
print (list(normalize('The eagle flies at midnight.')))

print (nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize("John Smith is from the United States of America and works at Microsoft Research Labs"))))

# import os
# from nltk.tag import StanfordNERTagger

# # change the paths below to point to wherever you unzipped the Stanford NER download file
# stanford_root = '/Users/benjamin/Development/stanford-ner-2014-01-04'
# stanford_data = os.path.join(stanford_root, 'classifiers/english.all.3class.distsim.crf.ser.gz')
# stanford_jar  = os.path.join(stanford_root, 'stanford-ner-2014-01-04.jar')

# st = StanfordNERTagger(stanford_data, stanford_jar, 'utf-8')
# for i in st.tag("John Bengfort is from the United States of America and works at Microsoft Research Labs".split()):
#     print '[' + i[1] + '] ' + i[0]

