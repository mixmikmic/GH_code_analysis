import nltk

# dictionary data type

pos = {}
pos

pos['colorless'] = 'ADJ'
pos

pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'

pos

pos['ideas']

pos['colorless']

pos['green']

list(pos)

sorted(pos)

[w for w in pos if w.endswith('s')]

for word in sorted(pos):
    print(word + ":", pos[word])

pos['sleep'] = ['N', 'V']

for word in sorted(pos):
    print(word + ":", pos[word])

pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}

pos

from collections import defaultdict
frequency = defaultdict(int)
frequency['coloreless'] = 4
frequency['ideas']

pos = defaultdict(list)
pos['sleep'] = ['NOUN', 'VERB']
pos['ideas']

pos = defaultdict(lambda: 'NOUN')
pos['colorless'] = 'ADJ'
pos['blog']

list(pos.items())

import nltk
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [word for (word, _) in vocab.most_common(1000)]
mapping = defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v
    
alice2 = [mapping[v] for v in alice]
print(alice2[:100])

len(set(alice2))

len(v1000)

len(set(alice))

from collections import defaultdict
counts = defaultdict(int)
from nltk.corpus import brown
for (word, tag) in brown.tagged_words(categories='news', tagset='universal'):
    counts[tag] += 1
counts['NOUN']

print(sorted(counts))

from operator import itemgetter
print( sorted(counts.items(), key=itemgetter(1), reverse=True) )

print( [t for t,c in sorted(counts.items(), key=itemgetter(1), reverse=True)] )

last_letters = defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)
print(last_letters['ly'])

print(last_letters['sy'])

anagrams = defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)

anagrams['aeilnrt']

anagrams = nltk.Index( (''.join(sorted(w)), w) for w in words)
anagrams['aeilnrt']

pos = defaultdict(lambda: defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
for ( (w1,t1), (w2, t2) ) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1 # tag and following word

pos[('DET', 'right')]

counts = defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word] += 1
    
[key for (key, value) in counts.items() if value == 32]

pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos2 = dict( (value, key) for (key, value) in pos.items() )
pos2['N']

pos.update({'cats': 'N', 'scratch': 'V', 'peacefully': 'ADV', 'old': 'ADJ'})
pos2 = defaultdict(list)
for key, value in pos.items():
    pos2[value].append(key)
    
pos2['ADV']

pos2 = nltk.Index((value, key) for (key, value) in pos.items())
pos2['ADV']



