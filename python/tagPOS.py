import itertools
import math
import re
import csv
import re;
import pandas as pd
import pylab as pyl
import nltk as nltk
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
#from nltk.tag.stanford import POSTagger
get_ipython().magic('matplotlib inline')
#enable longer display
pd.set_option('display.max_rows', 500)

d = (pd.read_csv('../../data/colorReference/message/colorReferenceMessage.csv', escapechar='\\'))
#     .query('sender == "speaker"'))

d['tokens'] = [[word for word in nltk.word_tokenize(sentence.lower()) if word.isalpha()]
               for sentence in d['contents']]

d['numWords'] = [pd.value_counts(words).sum() for words in d['tokens']]

d['pos'] = [[pos for (key, pos) in nltk.pos_tag(rowTokens)] 
            for rowTokens in d['tokens']]

d['numSuper'] = [sum([1 if label in ['JJS', 'RBS'] else 0 for label in posSet]) 
                 for posSet in d['pos']]
d['numComp'] = [sum([1 if label in ['JJR', 'RBR'] else 0 for label in posSet]) 
                 for posSet in d['pos']]

(d.drop(['tokens', 'pos'], 1)
  .to_csv("taggedColorMsgs.csv", index = False))

d.query("gameid == '3412-4'")



