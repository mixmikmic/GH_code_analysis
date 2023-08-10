from bs4 import BeautifulSoup
import urllib
import pdb
import json
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

train_facts = pickle.load(open('data/training_set.dat','r'))

test_facts = pickle.load(open('data/test_set.dat','r'))

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q) for fact,q in train_facts]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q) for fact,q in test_facts]


vocab = sorted(reduce(lambda x, y: x | y, (set(story + [answer]) for story, answer in train_stories + test_stories)))
story_vocab = sorted(reduce(lambda x, y: x | y, (set(story) for story, answer in train_stories + test_stories)))

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _ in train_stories + test_stories)))


print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')


answer_vocab = sorted(reduce(lambda x, y: x | y, (set([answer]) for _, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
answer_dict = dict((word, i) for i, word in enumerate(answer_vocab))
print('Answers dict len: {0}'.format(len(answer_dict)))

from scipy import stats
import numpy as np

lens = map(len, (x for x, _ in train_stories + test_stories))
print stats.describe(lens)
plt.xlabel('#words x story')
plt.hist(lens, bins=30,alpha=0.5)
plt.axvline(np.array(lens).mean(), color='black', linestyle='dashed', linewidth=2)
plt.savefig('plots/word_by_story.png')

lens = map(len, (x for x, _ in train_facts + test_facts))
print stats.describe(lens)
plt.xlabel('#facts')
plt.hist(lens, bins=30,alpha=0.5)
plt.axvline(np.array(lens).mean(), color='black', linestyle='dashed', linewidth=2)
plt.savefig('plots/facts_by_disease.png')

lens = map(len, (x for h,_ in train_facts + test_facts for x in h))
print lens[0]
print stats.describe(lens)
plt.xlabel('#words x facts')
plt.hist(lens, bins=30,alpha=0.5)
plt.axvline(np.array(lens).mean(), color='black', linestyle='dashed', linewidth=2)
plt.savefig('plots/word_by_fact.png')

#print story_vocab



