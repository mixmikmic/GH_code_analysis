import nltk

text = u'Gerd suchte ca. 5 min. die 3 Freunde bzw. Kollegen. Sie warteten am 1. Mai in Berlin/ West: am Zoo.'

from nltk.tokenize import sent_tokenize

# install punkt sentence tokenizer
nltk.download('punkt')

# seems to be unnecessary: loading the german punkt tokenizer explicitly
#tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')

# split sentences
# NLTK has problems with ca., bzw.

sentences = sent_tokenize(text, language='german')
#sentences = tokenizer.tokenize(text)

for i, s in enumerate(sentences):
    print(i+1, '-->', s)

from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')

def pos2string(tagged): return ' '.join(['/'.join(p) for p in tagged])

# tag each word in every sentence
for i, s in enumerate(sentences):
    tagged = nltk.pos_tag(word_tokenize(s, language='german'))
    print(i+1, '-->', pos2string(tagged))

def pos_filter(tagged, type = 'NN'): return [x[0] for x in tagged if x[1].startswith(type)]

print('Nouns:')
for i, s in enumerate(sentences):
    print(i+1, '-->', pos_filter(nltk.pos_tag(word_tokenize(s, language='german')), 'NN'))

print('Verbs:')
for i, s in enumerate(sentences):
    print(i+1, '-->', pos_filter(nltk.pos_tag(word_tokenize(s, language='german')), 'VB'))

from nltk import ne_chunk, pos_tag

nltk.download('maxent_ne_chunker')
nltk.download('words')

def ner_tag(chunked): return [c[0]+'/0' if type(c) == tuple else c.leaves()[0][0]+'/'+c.label() for c in chunked]

for i, s in enumerate(sentences):
    chunked = ne_chunk(pos_tag(word_tokenize(s, language='german')))
    print(i+1, '-->', ' '.join(ner_tag(chunked)))

