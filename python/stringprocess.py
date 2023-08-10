import itertools

def rle(s):
    res = [(c,len(list(gen))) for c,gen in itertools.groupby(s)]
    return res

rle('aaaaabbbbbaaaaacccccbbbb')

from collections import defaultdict

Tree = lambda: defaultdict(Tree)

T = Tree()
T['tamil']['yuva']=12
T['tamil']['tharani'] = 14

print(T)

import pprint
pprint.pprint(T)

import regex

regex.split('(?V1)(-)','a-beautiful-day-papa')

def tsplit(s,sep=' '):
    res = []
    iidx = 0
    for ci,c in enumerate(s):
        if c == sep:
            res += [s[iidx:ci]]
            iidx = ci+1
    res += [s[iidx:]]
    return res

