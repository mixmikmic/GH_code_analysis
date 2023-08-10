import sys
import collections
import subprocess

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

API = fabric.load('etcbc4b', 'lexicon', 'movement', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads
        function
        lex g_word_utf8 trailer_utf8 prs uvf sp gloss
        book chapter verse label number
    ''',''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

nlclauses = 0
lclauses = set()
mclauses = set()
lverbs = {}
mverbs = {}
freq = collections.Counter()
contxt = collections.defaultdict(lambda: [])

for p in F.otype.s('phrase'):
    if F.function.v(p) == 'Pred':
        nlclauses += 1
        c = L.u('clause', p)
        is_move = False
        is_move_x = False
        transitive = False
        for v in L.d('word', c):
            if F.uvf.v(v) == 'H':
                is_move = True
        for pp in L.d('phrase', c):
            if F.function.v(pp) == 'Loca':
                lclauses.add(c)
                if is_move:
                    mclauses.add(c)
                    is_move_x = True
            elif F.function.v(pp) == 'Objc':
                transitive = True
        if is_move_x:
            for v in L.d('word', p):
                if F.sp.v(v) == 'verb':
                    lex = F.lex.v(v)
                    gloss = F.gloss.v(v)
                    lverbs[lex] = gloss
                    if is_move: mverbs[lex] = F.gloss.v(v)
                    freq[lex] += 1
                    bk = F.book.v(L.u('book', v))
                    ch = F.chapter.v(L.u('chapter', v))
                    vs = F.verse.v(L.u('verse', v))
                    txt = ''.join('{}{}'.format(F.g_word_utf8.v(w), F.trailer_utf8.v(w)) for w in L.d('word', c))
                    contxt[lex].append((txt, 'T' if transitive else 'I', lex, gloss, vs, ch, bk))
                    
print('{} verbs of motion of {} verbs of location of {} clauses with a predicate'.format(
    len(mverbs), len(lverbs), nlclauses))

of = outfile('lverbs.csv')
for v in sorted(lverbs, key=lambda x: (-freq[x], x)):
    of.write('{},{},{}\n'.format(v, lverbs[v], freq[v]))
of.close()
of = outfile('mverbs.csv')
for v in sorted(mverbs, key=lambda x: (-freq[x], x)):
    of.write('{},{},{}\n'.format(v, mverbs[v], freq[v]))
of.close()
of = outfile('context.csv')
for v in sorted(contxt, key=lambda x: (-freq[x], x)):
    for occ in contxt[v]:
        of.write('{}\n'.format('{},{},{},{},{},{},{}'.format(*occ).replace('\n', ' ')))
of.close()

get_ipython().system("head -n 20 {my_file('mverbs.csv')}")



