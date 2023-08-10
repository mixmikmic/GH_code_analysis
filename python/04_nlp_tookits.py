from __future__ import unicode_literals

import nltk

nltk.word_tokenize(u'Geladeira Brastemp c/ painel branco-amarelo. Atualização em (09/12/2015).', language='portuguese')

from nltk.corpus import mac_morpho

mac_morpho.tagged_words()

from nltk.corpus import floresta

floresta.parsed_sents()

from pattern.en import tag

s = "I eat pizza with a fork."
s = tag(s)
print s
for word, tag in s:
    if tag == "NN": # Find all nouns in the input string.
        print word

from pattern.web import Twitter

twitter = Twitter(language='pt')

for tweet in twitter.search('Dilma', cached=False):
    print tweet.text

from textblob import TextBlob
text = 'Geladeira Brastemp 2 Portas Branca com sensor de porta aberta'
blob = TextBlob(text)

blob.detect_language()

blob.tags

blob.noun_phrases

# path used for polyglot downloaded data
import polyglot
polyglot.data_path = '/usr/share/'

from polyglot.downloader import downloader
downloader.supported_tasks(lang="pt")

from polyglot.text import Text, Word

text = Text(u'Geladeira Brastemp c/ painel branco-amarelo. Atualização em (09/12/2015).')

text.detect_language()

print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
for word, tag in text.pos_tags:
    print(u"{:<16}{:>2}".format(word, tag))

word = Word(u'geladeira', language="pt")
print("Neighbors (Synonms) of {}".format(word)+"\n"+"-"*30)
for w in word.neighbors:
    print("{:<16}".format(w))
print("\n\nThe first 10 dimensions out the {} dimensions\n".format(word.vector.shape[0]))
print(word.vector[:10])

word = Word(u'infelicidade')
word.morphemes

text = Text('Apesar de interessante, o produto é caro e deixa a desejar. Não recomendo')
print("{:<16}{}".format("Word", "Polarity")+"\n"+"-"*30)
for w in text.words:
    print("{:<16}{:>2}".format(w, w.polarity))

import nlpnet
nlpnet.set_data_dir(str('/usr/share/nlpnet_data/'))
tagger = nlpnet.POSTagger()
tagger.tag(u'Geladeira Brastemp c/ painel branco-amarelo. Atualização em (09/12/2015).')

tagger = nlpnet.SRLTagger()
sent = tagger.tag(u'O rato roeu a roupa do rei de Roma em abril.')[0]
sent.arg_structures



