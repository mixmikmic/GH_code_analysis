import json
from pycorenlp import StanfordCoreNLP

text = u'Gerd suchte ca. 5 min. die 3 Freunde bzw. Kollegen. Sie warteten am 1. Mai in Berlin/ West: am Zoo.'

# create annotator
# 'localhost' does not work inside container - use local ip address
corenlp_server = 'http://192.168.178.20:9000/'
props = {'annotators': 'tokenize,ssplit,pos,ner'}

nlp = StanfordCoreNLP(corenlp_server)

# annotate text
result = nlp.annotate(text, properties=props)

res = json.loads(result, encoding='utf-8', strict=True)

# CoreNLP has problems with abbreviations (when followed by '/')

# split sentences
for i, s in enumerate(res['sentences']):
    print(i+1, '-->', ' '.join([t['word'] for t in s['tokens']]))

# print POS tags
for i, s in enumerate(res['sentences']):
    print(i+1, '-->', ' '.join([t['word']+'/'+t['pos'] for t in s['tokens']]))

def pos_filter(tokens, type='NN'):
    return [t for t in tokens if t['pos'].startswith(type)]

print('Nouns:')
for i, s in enumerate(res['sentences']):
    print(i+1, '-->', ' '.join([t['word']+'/'+t['pos'] for t in pos_filter(s['tokens'], 'NN')]))

print('Verbs:')
for i, s in enumerate(res['sentences']):
    print(i+1, '-->', ' '.join([t['word']+'/'+t['pos'] for t in pos_filter(s['tokens'], 'VV')]))

# print NER tags
for i, s in enumerate(res['sentences']):
    print(i+1, '-->', ' '.join([t['word']+'/'+t['ner'] for t in s['tokens']]))



