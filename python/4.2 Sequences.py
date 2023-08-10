t = 'walk', 'fem', 3
t

t[0]

t[1:]

len(t)

raw = 'I turned off the spectroroute'
text = ['I', 'turned', 'off', 'the', 'spectroroute']
pair = (6, 'turned')
raw[2], text[3], pair[1]

raw[-3:], text[-3:], pair[-3:]

len(raw), len(text), len(pair)

import nltk
from nltk import word_tokenize

raw = 'Red lorry, yellow lorry, red lorry, yellow lorry.'
text = word_tokenize(raw)
fdist = nltk.FreqDist(text)
sorted(fdist)
# FreqDist converted into a list, using sorted

for key in fdist:
    print(key + ':', fdist[key], end='; ')

words = ['I', 'turned', 'off', 'the', 'spectroroute']
words[2], words[3], words[4] = words[3], words[4], words[2]
words

tmp = words[2]
words[2] = words[3]
words[3] = words[4]
words[4] = tmp

words

words = ['I', 'turned', 'off', 'the', 'spectroroute']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
zip(words, tags)

list(zip(words, tags))

list(enumerate(words))

text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
text == training_data + test_data

len(training_data) / len(test_data)

words = 'I turned off the spectroroute'.split()
wordlens = [(len(word), word) for word in words]
# tuple (len(word), word)
wordlens.sort()
' '.join(w for (_, w) in wordlens)

lexicon = [
    ('the', 'det', ['Di:', 'D@']),
    ('off', 'prep', ['Qf', 'O:f'])
]
# list, because it is a collection of objects of a single type

# position not significant -> lists

#lists are mutable
lexicon.sort()
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
del lexicon[0]
#tuples are inmutables

lexicon = tuple(lexicon)
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
del lexicon[0]

text = '''"When I use a word," Humpty Dumpty said in rather a scornful tone,
"it means just what I choose it to mean - neither more nor less."'''
[w.lower() for w in word_tokenize(text)]

max([w.lower() for w in word_tokenize(text)])
# storage is necessary for the list, before max is called

max(w.lower() for w in word_tokenize(text))
# the data is streamed to the calling function

min(w.lower() for w in word_tokenize(text))



