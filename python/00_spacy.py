from __future__ import unicode_literals # If Python 2
import spacy.en
from spacy.tokens import Token
from spacy.parts_of_speech import ADV

nlp = spacy.en.English()

# Find log probability of Nth most frequent word
probs = [lex.prob for lex in nlp.vocab]
probs.sort()
words = [w for w in nlp.vocab if hasattr(w,'repvec')]#if w.has_repvec]

tokens = nlp(u'"I ran to the wall quickly," Frank explained to the robot.')
ran = tokens[2]
quickly = tokens[6]
run = nlp(ran.lemma_)[0]

# the integer and string representations of "moved" and its head
print (ran.orth, ran.orth_, ran.head.lemma, ran.head.lemma_)
print (quickly.orth, quickly.orth_, quickly.lemma, quickly.lemma_,)
print (quickly.head.orth_, quickly.head.lemma_)
print (ran.prob, run.prob, quickly.prob)
print (ran.cluster, run.cluster, quickly.cluster)

is_adverb = lambda tok: tok.pos == ADV and tok.prob < probs[-100]
str_ = u'"I ran to the wall quickly," Frank explained to the robot.'
tokens = nlp(str_)
print u''.join(tok.string.upper() if is_adverb(tok) else tok.string for tok in tokens)
quickly = tokens[6]

from numpy import dot
from numpy.linalg import norm

cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
words.sort(key=lambda w: cosine(w.vector, quickly.vector))
words.reverse()

print('1-20:')
print('\n'.join(w.orth_ for w in words[0:20]))
print('\n50-60:')
print('\n'.join(w.orth_ for w in words[50:60]))
print('\n100-110:')
print('\n'.join(w.orth_ for w in words[100:110]))
print('\n1000-1010:')
print('\n'.join(w.orth_ for w in words[1000:1010]))
print('\n50000-50010:')
print('\n'.join(w.orth_ for w in words[50000:50010])) 

say_adverbs = ['quickly', 'swiftly', 'speedily', 'rapidly']
say_vector = sum(nlp.vocab[adverb].repvec for adverb in say_adverbs) / len(say_adverbs)
words.sort(key=lambda w: cosine(w.repvec, say_vector))
words.reverse()
print('1-20:')
print('\n'.join(w.orth_ for w in words[0:20]))
print('\n50-60:')
print('\n'.join(w.orth_ for w in words[50:60]))
print('\n1000-1010:')
print('\n'.join(w.orth_ for w in words[1000:1010]))

from spacy.parts_of_speech import NOUN

is_noun = lambda tok: tok.pos == NOUN and tok.prob < probs[-1000]
print u''.join(tok.string.upper() if is_noun(tok) else tok.string for tok in tokens)
nouns = [tok for tok in tokens if is_noun(tok)]

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

barrier = nlp('barrier')[0]
car = nlp('car')[0]
agent = nlp('android')[0]
test_nouns = nouns + [barrier] + [car] + [agent]

n = len(test_nouns)
barrier_relations = np.zeros(n)
car_relations = np.zeros(n)
agent_relations = np.zeros(n)
for i, noun in enumerate(test_nouns):
    barrier_relations[i] = cosine(barrier.repvec, noun.repvec)
    car_relations[i] = cosine(car.repvec, noun.repvec) 
    agent_relations[i] = cosine(agent.repvec, noun.repvec) 


fig, ax = plt.subplots(figsize=(10,8))

index = np.arange(n)
bar_width = 0.2

opacity = 0.4

rects1 = plt.bar(index, barrier_relations, bar_width,
                 alpha=opacity,
                 color='b',
                 label=barrier.orth_)

rects2 = plt.bar(index + bar_width, car_relations, bar_width,
                 alpha=opacity,
                 color='r',
                 label=car.orth_)

rects3 = plt.bar(index + 2 * bar_width, agent_relations, bar_width,
                 alpha=opacity,
                 color='g',
                 label=agent.orth_)

labels = [tok.orth_ for tok in test_nouns]

plt.xlabel('Test Word')
plt.ylabel('Similarity')
plt.title('Similarity of words')
plt.xticks(index + bar_width, labels)
plt.legend()

from IPython.core.display import HTML

# Borrowed style from Probabilistic Programming and Bayesian Methods for Hackers
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()



