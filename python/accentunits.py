import sys, os

import laf
from laf.fabric import LafFabric
#from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
API = fabric.load('etcbc{}'.format(version), 'lexicon,para', 'paragraphs', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype monads
        g_word_utf8 trailer_utf8
    ''',
    '''
    '''),
#    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

inf('Compiling index of accented units ...')
word2au = {}     # the mapping from word node to accent unit
aus = set()      # only needed to count the total number of accented units
glue = {'', '־'}  # the interword material that continues the current au
current_au = []
for w in F.otype.s('word'):
    current_au.append(w)
    word2au[w] = current_au
    if F.trailer_utf8.v(w) not in glue: # move to a new au
        aus.add(tuple(current_au))
        current_au = []
if current_au: aus.add(tuple(current_au))
inf('Assembled {} words into {} accented units'.format(
    len(word2au.keys()),
    len(aus),
))        
    

text = ''
verse = 0
for n in NN():
    otype = F.otype.v(n)
    if otype == 'verse':
        verse += 1
        if verse > 11: break
        text += '\nGenesis 1:{}\n'.format(verse)
        prev_au = None
    elif otype == 'word':
        this_au = word2au[n]
        if prev_au != None and this_au is not prev_au:
            text += ' ({}) '.format(','.join(F.monads.v(x) for x in prev_au))
        prev_au = this_au
        text += F.g_word_utf8.v(n)+F.trailer_utf8.v(n)
if prev_au:
    text += ' ({}) '.format(','.join(F.monads.v(x) for x in prev_au))

print(text)



