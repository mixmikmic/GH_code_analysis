import sys
import collections
import subprocess

from lxml import etree

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.mql import MQL
fabric = LafFabric()

API = fabric.load('etcbc4', '--', 'tabtxt', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads
        tab txt 
        book chapter verse
    ''','''
    '''),
    "prepare": prepare,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))
Q = MQL(API)

query = '''
select all objects where
[book [chapter [verse
[clause
    [clause_atom]
]
]]]
'''
sheaf = Q.mql(query)

outf = outfile('tabtxt.txt')
maxtxt = 0
maxtab = 0
for ((bk, ((ch, ((vs, ((c, ((ca,),)),)),)),)),) in sheaf.results():
    txt = F.txt.v(c)
    tab = F.tab.v(ca)
    ntxt = len(txt)
    ntab = int(tab)
    if ntxt > maxtxt: maxtxt = ntxt
    if ntab > maxtab: maxtab = ntab
    passage = '{:<20} {:>3}:{:>3}'.format(F.book.v(bk), F.chapter.v(ch), F.verse.v(vs))
    outf.write('{:<28} {:>10}={:<10}\n'.format(passage, txt, tab))
outf.close()
print('max TXT = {}; max TAB = {}'.format(maxtxt, maxtab))



