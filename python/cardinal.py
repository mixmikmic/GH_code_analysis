import sys
import collections
import subprocess

from lxml import etree

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.mql import MQL
fabric = LafFabric()

API = fabric.load('etcbc4', '--', 'mql', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads
        lex_utf8
        sp typ vs function
        book chapter verse label
    ''','''
        functional_parent
    '''),
    "prepare": prepare,
    "primary": True,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))
Q = MQL(API)

card_query = '''
select all objects where
[book [chapter [verse
[clause
	[phrase
		[word sp = verb]
	]
	..
	[phrase
		[word FIRST AND LAST ls  = card AND prs = "absent"]
	]
] 
OR
[clause
	[phrase
		[word FIRST AND LAST ls  = card AND prs = "absent"]
	]
	..
	[phrase
		[word sp = verb]
	]
]
]]]
'''

sheaf = Q.mql(card_query)

for x in sheaf.results():
    print(x)
    break

tsvfile = outfile('card.csv')
nresults = 0
tsvfile.write('book\tchapter\tverse\torder\tclause typ\tverbal stem\tverb\tcard function\tcard\tphrase verb text\tphrase card text\tclause text\n')
for ((book, ((chapter, ((verse, ((clause, ((phrase1, ((word1,),)), (phrase2, ((word2,),)))),)),)),)),) in sheaf.results():
    nresults += 1
    verb_first = F.sp.v(word1) == 'verb'
    word_verb = word1 if verb_first else word2
    word_card = word2 if verb_first else word1
    phrase_verb = phrase1 if verb_first else phrase2
    phrase_card = phrase2 if verb_first else phrase1
    clause_text = ' ~ '.join(x[1] for x in P.data(clause))
    phrase_verb_text = ' ~ '.join(x[1] for x in P.data(phrase_verb))
    phrase_card_text = ' ~ '.join(x[1] for x in P.data(phrase_card))
    
    tsvfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        F.book.v(book), 
        F.chapter.v(chapter), 
        F.verse.v(verse),
        '>>' if verb_first else '<<',
        F.typ.v(clause),
        F.vs.v(word_verb),
        F.lex_utf8.v(word_verb),
        F.function.v(phrase_card),
        F.lex_utf8.v(word_card),
        phrase_verb_text,
        phrase_card_text,
        clause_text,
    ))
tsvfile.close()
print('There are {} results'.format(nresults))

get_ipython().system("head -n 10 {my_file('card.csv')}")



