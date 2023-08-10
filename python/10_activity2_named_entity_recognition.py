from __future__ import unicode_literals
import codecs

# this could be done in a iterate way for performance in huge corpus
with codecs.open('corpus.txt', encoding='utf8') as fp:
    corpus = fp.read()
    
# corpus has 55 millions characters. I am going to use just 10k to speed up a bit the things
corpus = corpus[:10000]

import polyglot
polyglot.data_path = '/usr/share/'

from polyglot.text import Text
text = Text("A presidenta do Brasil é Dilma Roussef.")

text.entities

from polyglot.text import Text
text = Text(corpus[:2000])
# check with more than 2k. Polyglot has a bug with unicode text

text.entities

# first we need to tag the corpus.
import nlpnet
nlpnet.set_data_dir(str('/usr/share/nlpnet_data/'))
tagger = nlpnet.POSTagger()
sentences = tagger.tag(corpus)

sentences[0]

import nltk
grammar = "NE: {<NPROP>+}"
cp = nltk.RegexpParser(grammar)

sentence = sentences[0]

sentence

print cp.parse(sentence)

entities = set()
for tree in cp.parse_sents(sentences):
    for subtree in tree.subtrees():
        if subtree.label() == 'NE': 
            entity = ' '.join([word for word, tag in subtree.leaves()])
            entities.add(entity)
            
from pprint import pprint
pprint(entities)



