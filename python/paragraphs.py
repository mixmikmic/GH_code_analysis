import sys, os
import collections

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
API = fabric.load('etcbc{}'.format(version), 'lexicon,para', 'paragraphs', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype 
        gloss
        pargr
    ''',
    '''
    '''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

limit = 100
n = 0
for ca in F.otype.s('clause_atom'):
    n += 1
    if n > limit: break
    inf('{:>10} {}'.format(
        F.pargr.v(ca),
        ' '.join(F.gloss.v(w) for w in L.d('word', ca)),
    ), withtime=False)



