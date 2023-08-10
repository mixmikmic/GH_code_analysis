import sys
import collections
import subprocess

from lxml import etree

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.mql import MQL
fabric = LafFabric()

API = fabric.load('etcbc4b', '--', 'ntn', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads number
        language lex g_word
        sp gn nu ps vt vs
        function typ det
        book chapter verse label
    ''',''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))
Q = MQL(API)

test_query = '''
select all objects where
[book book = Genesis
[chapter chapter = 45
[verse verse = 22
[sentence number = 73
[clause number = 1
    [phrase focus first and last] or [phrase focus first] [[phrase focus][gap?]]* [phrase focus last]
]
]]]]
'''
test_sheaf = Q.mql(test_query)
print('{} results'.format(test_sheaf.nresults()))
for x in test_sheaf.results():
    print(x)

ntn_sim_query = '''
select all objects where

[book[chapter[verse
[clause
 [phrase function = Pred 
     [word focus first (language = Hebrew and (lex = 'NTN[' or lex = 'FJM[')) ]
 ]
]
]]]
'''
clause_query = '''
select all objects where
[clause
    [word focus first and last] or [word focus first] [[word focus][gap?]]* [word focus last]
]
'''
clause_p_query = '''
select all objects where
[clause
    [phrase focus]
]
'''

sheaf_ntn_sim = Q.mql(ntn_sim_query)
print('{} results'.format(sheaf_ntn_sim.nresults()))
for x in sheaf_ntn_sim.results():
    print(x)
    break

sheaf_clause = Q.mql(clause_query)
print('{} results'.format(sheaf_clause.nresults()))

sheaf_clause_p = Q.mql(clause_p_query)
print('{} results'.format(sheaf_clause_p.nresults()))

verb_index = {}
phrase_index = {}
word_index = {}
book_index = {}
passage_index = {}
relevant_clauses = set()
doubles = 0
for ((book, ((chapter, ((verse, ((clause, ((phrase, ((word,),)),)),)),)),)),) in sheaf_ntn_sim.results():
    if clause in relevant_clauses: doubles += 1
    relevant_clauses.add(clause)
    book_index[clause] = F.book.v(book)
    passage_index[clause] = '{} {}:{}#{}'.format(
        F.book.v(book), 
        F.chapter.v(chapter), 
        F.verse.v(verse), 
        clause,
    )
    wlex = F.lex.v(word)
    phrase_index[phrase] = wlex
    word_index[word] = wlex
nclauses = len(relevant_clauses)
print('{} clauses of which {} duplicates => total {}'.format(
    nclauses, doubles, nclauses - doubles,
))
print('{} phrases in index\n{} words in index'.format(len(phrase_index), len(word_index)))

outf = outfile('ntn_sim.csv')
nwritten = 0
tclauses = 0
clauses_w = set()
still_in_pre = 0
for ((clause, words),) in sheaf_clause.results():
    tclauses += 1
    clauses_w.add(clause)
    if clause not in relevant_clauses: continue
    pre_info = set()
    post_info = set()
    verb_info = ''
    in_pre = True
    for word in words:
        word = word[0]
        this_verb_info = word_index.get(word, None)
        if this_verb_info != None:
            in_pre = False
            verb_info += this_verb_info
        else:
            info = {
                F.lex.v(word).replace('_','-'),
                F.sp.v(word), 
                F.gn.v(word), 
                F.nu.v(word), 
                F.ps.v(word), 
                F.vt.v(word), 
                F.vs.v(word),
            } - {
                'NA', 
                'unknown',
            }
            if in_pre:
                pre_info |= info
            else:
                post_info |= info
    if in_pre: still_in_pre += 1
    outf.write('{},{},{},{}\n'.format(
        passage_index[clause], 
        verb_info, 
        '_'.join(pre_info | post_info), 
        book_index[clause],
    ))
    nwritten += 1
outf.close()
print('{} clauses, {} lines written'.format(tclauses, nwritten))
print('{} still in pre'.format(still_in_pre))

outfp = outfile('ntn_sim_p.csv')
outfpo = outfile('ntn_sim_po.csv')
nwritten = 0
tclauses = 0
clauses_p = set()
still_in_pre = 0
curclause = None
i = 0
for ((clause, ((phrase,),)),) in sheaf_clause_p.results():
    if clause != curclause:
        if curclause != None:
            if in_pre: still_in_pre += 1
            outfp.write('{},{},{}\n'.format(passage_index[curclause], verb_info, '_'.join(pre_info | post_info)))
            outfpo.write('{},{},{}\n'.format(passage_index[curclause], verb_info, '_'.join(pre_info_o | post_info_o)))
            nwritten += 1
        tclauses += 1
        clauses_p.add(clause)
        if clause not in relevant_clauses:
            curclause = None
            continue
        curclause = clause
        pre_info = set()
        pre_info_o = set()
        post_info = set()
        post_info_o = set()
        verb_info = ''
        in_pre = True
        i = 0
    else:
        i += 1
    this_verb_info = phrase_index.get(phrase, None)
    if this_verb_info != None:
        in_pre = False
        verb_info += this_verb_info
    else:
        info = {
            F.function.v(phrase), 
            F.typ.v(phrase), 
            F.det.v(phrase),
        } - {
            'NA', 
            'unknown',
        }
        info_o = {'{}{}'.format(i, v) for v in info}
        if in_pre:
            pre_info |= info
            pre_info_o |= info_o
        else:
            post_info |= info
            post_info_o |= info_o
if curclause != None:
    if in_pre: still_in_pre += 1
    outfp.write('{},{},{}\n'.format(passage_index[curclause], verb_info, '_'.join(pre_info | post_info)))
    outfpo.write('{},{},{}\n'.format(passage_index[curclause], verb_info, '_'.join(pre_info_o | post_info_o)))
    nwritten += 1
outfp.close()
outfpo.close()
print('{} clauses, {} lines written'.format(tclauses, nwritten))
print('{} still in pre'.format(still_in_pre))





