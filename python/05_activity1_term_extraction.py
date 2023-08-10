from __future__ import unicode_literals

import codecs

with codecs.open('corpus.txt','r',encoding='utf8') as fp:
    corpus = fp.read()

len(corpus)

corpus = corpus[:1000000]

len(corpus)

import nlpnet
nlpnet.set_data_dir(b'/usr/share/nlpnet_data/')
tagger = nlpnet.POSTagger()
sentences = tagger.tag(corpus)
# tagset: http://nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf

len(sentences)

sentences[2]

import enchant
d = enchant.Dict("pt_BR")

freq_list = dict()
for sentence in sentences:
    chunk = ''
    for word, tag in sentence:
        word = word.lower()
        if tag == 'N':
            if not chunk:
                chunk = word
            else:
                chunk += ' ' + word
        else:
            if chunk:
                if d.check(chunk):
                    freq_list[chunk] = freq_list.get(chunk, 0) + 1
    if chunk:
        if d.check(chunk):
            freq_list[chunk] = freq_list.get(chunk, 0) + 1

from operator import itemgetter
terms = sorted(freq_list.items(), key=itemgetter(1), reverse=True)
print 'Best Terms:' + ','.join([word for word,tag in terms][:20])



