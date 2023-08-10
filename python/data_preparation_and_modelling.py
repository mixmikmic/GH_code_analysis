import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import random

data = pd.read_json('./cerpenmu/output.json')

for d in data['text'][:3]:
    print(d)
    print('')

sentence_list = []
for d in data['text']:
    d = re.sub(r'<br>', '', d)
    d = re.sub(r'<[\/]{,1}p>', '', d)
    d = re.sub(r'“', '', d)
    d = re.sub(r'"', '', d)
    d = d.lower()
    for s in sent_tokenize(d):
        sentence_list.append(s)

word_list = [word_tokenize(s) for s in sentence_list]

import gensim

model = gensim.models.Word2Vec(word_list, size=100, workers=4)  # an empty model, no training yet
model.train(word_list, total_examples=model.corpus_count, epochs=100)  # can be a non-repeatable, 1-pass generator

model.most_similar(positive=['jakarta'], topn=20)



