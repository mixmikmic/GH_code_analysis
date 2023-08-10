import sys, os, csv, re
import collections
import subprocess

from lxml import etree
from pprint import pprint

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.lib import Transcription, monad_set

#from etcbc.mql import MQL
fabric = LafFabric()

API=fabric.load('etcbc4b', '--', 'mql', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype nu ps gn vs vt prs ls lex g_cons g_word_utf8 g_cons_utf8
        function txt
        book chapter verse label sp kind typ 
    ''',
    ''' functional_parent
    '''),
    "prepare": prepare,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))
#Q = MQL(API)

infc_lst = []


def find_infc():
    '''Finds the infinitive construct phrase with verbal stem hifil or nifal with prepostion b.'''
    for node in NN():
        if F.otype.v(node) == 'word': 
            if F.vt.v(node) == 'infc' and F.vs.v(node) == 'hif' or F.vs.v(node) == 'nif':
                phrases = L.u('phrase', node)
                b_found = False
                for word in L.d('word', phrases):
                    if word == node and b_found:   
                        infc_lst.append(word)
                        break
                    b_found = F.lex.v(word) == 'B'

            
find_infc()

words = [F.g_cons_utf8.v(n) for n in infc_lst]
len(words)

def is_start(n):
    '''Finds the second word in the clause. The infc is in the second position of the clause'''
    c = L.u('clause', n)
    words = list(L.d('word', c))
    return len(words) > 1 and words[1] == n

data_lst = []
for n in infc_lst: 
    book = L.u('book', n)
    verse = L.u('verse', n)
    clause = L.u('clause', n)
    data_lst.append([
            str(n), 
            str(is_start(n)), 
            F.g_cons.v(n), 
            T.book_name(book), 
            str(F.chapter.v(verse)), 
            str(F.verse.v(verse)),
            T.words(L.d('word', clause)).strip('\n')
    ])

with open("b-Inf.csv", 'w') as f:
    header = ['node', 'start', 'verb', 'book', 'chapter', 'verse', 'clause']
    f.write('{}\n'.format(','.join(header)))

    for item in data_lst:
        f.write('{}\n'.format(','.join(item)))

for (node, start, verb, book, chapter, verse, clause) in data_lst:
    print('{} {} {}'.format(verb, start, clause))

