import sys, os

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
API = fabric.load('etcbc{}'.format(version), 'lexicon,para', 'paragraphs', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype 
    ''',
    '''
    '''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

inf('Inspecting all clauses')
chapter_cross = set()
verse_cross = set()
for c in F.otype.s('clause'):
    words = L.d('word', c)
    (first_word, last_word) = (words[0], words[-1])
    if L.u('chapter', first_word) != L.u('chapter', last_word):
        chapter_cross.add(c)
    if L.u('verse', first_word) != L.u('verse', last_word):
        verse_cross.add(c)
inf('Done')
print('# chapter crossing clauses: {:>4}'.format(len(chapter_cross)))
print('# verse   crossing clauses: {:>4}'.format(len(verse_cross)))

for c in verse_cross:
    print('{}\n{}\n'.format(T.passage(c), T.words(L.d('word', c)).replace('\n', ' ')))



