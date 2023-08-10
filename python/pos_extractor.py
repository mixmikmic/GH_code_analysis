import json
from pycorenlp import StanfordCoreNLP

# 'localhost' does not work inside container - use local ip address
corenlp_server = 'http://192.168.178.20:9000/'

nlp = StanfordCoreNLP(corenlp_server)

import pandas as pd

# read data
data_file = 'data/documents.csv.gz'
data = pd.read_csv(data_file, delimiter='\t', quoting=3, names = ('id', 'text'))

print('loaded {} data entries.'.format(len(data)))

NN_counts = {}
VV_counts = {}

import sys

def pos_filter(sentences, type='NN'):
    return [t['word'] for s in sentences for t in s['tokens'] if t['pos'].startswith(type)]

def count_item(counts, item):
    assert type(counts) == dict
    if not item in counts:
        counts[item] = 1
    else:
        counts[item] += 1

props = {'annotators': 'tokenize,ssplit,pos'}

length = len(data)
for i in range(0,length):
    if (i % 100 == 0):
        print("step: ", i)
    
    text = data['text'][i]
    
    result = nlp.annotate(text, properties=props)
    res = json.loads(result, encoding='utf-8', strict=True)
    
    for noun in pos_filter(res['sentences'], 'NN'): count_item(NN_counts, noun)
    for verb in pos_filter(res['sentences'], 'VV'): count_item(VV_counts, verb)

print("#nouns: ", len(NN_counts))
print("#verbs: ", len(VV_counts))

NN_sorted = sorted(NN_counts.items(), key=lambda t: (t[1],t[0]), reverse=True)
VV_sorted = sorted(VV_counts.items(), key=lambda t: (t[1],t[0]), reverse=True)

print('->Top nouns:')
for key, value in NN_sorted[0:9]:
    print("%s: %s" % (key, value))
print()
print('->Top verbs:')
for key, value in VV_sorted[0:9]:
    print("%s: %s" % (key, value))

import csv

def save_counts(csv_file, counts):
    with open(csv_file, 'w') as out:
        w = csv.writer(out)
        for row in counts:
            w.writerow((row[0], row[1]))
            
save_counts('count_nouns.csv', NN_sorted)
save_counts('count_verbs.csv', VV_sorted)



