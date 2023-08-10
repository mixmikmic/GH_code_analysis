import sys, os
import collections

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
API = fabric.load('etcbc{}'.format(version), 'lexicon', 'adjectives', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype 
        function rela sp
        gloss
        g_word_utf8 trailer_utf8
        book chapter verse number
    ''',
    '''
        mother
    '''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

attr_subphrases = set()
inf('Finding subphrases ...')
for s in F.otype.s('subphrase'):
    if F.rela.v(s) != 'atr':
        continue
    attr_subphrases.add(s)
inf('{} attributive subphrases'.format(len(attr_subphrases)))

attr_subphrase_mother = dict()
multiple_mothers = set()
no_mothers = set()
for s in attr_subphrases:
    mothers = list(C.mother.v(s))
    if len(mothers) == 0:
        no_mothers.add(s)
        continue
    if len(mothers) > 1: 
        multiple_mothers.add(s)
        continue
    attr_subphrase_mother[s] = mothers[0]
if len(multiple_mothers):
    msg('{} subphrases with multiple mothers'.format(len(multiple_mothers)))
else:
    inf('No subphrases with multiple mothers')
if len(no_mothers):
    msg('{} subphrases without mothers'.format(len(no_mothers)))
else:
    inf('No subphrases without mothers')

inf('{} attributive subphrases with a single mother'.format(len(attr_subphrase_mother)))

mother_types = collections.Counter()
idents = 0
for (s, m) in attr_subphrase_mother.items():
    mother_types[F.otype.v(m)] +=1

for t in sorted(mother_types):
    print('{:>4} subphrases with a mother of type {}'.format(mother_types[t], t))

mother_length = collections.Counter()
for (s, m) in attr_subphrase_mother.items():
    mother_length[len(L.d('word', m))] +=1

for t in sorted(mother_length):
    print('{:>4} subphrases with a mother of length {:>2}'.format(mother_length[t], t))

mother_nouns = collections.Counter()
for (s, m) in attr_subphrase_mother.items():
    mother_nouns[len([w for w in L.d('word', m) if F.sp.v(w) == 'subs'])] +=1

for t in sorted(mother_nouns):
    print('{:>4} subphrases with a mother having {:>2} nouns'.format(mother_nouns[t], t))

fields = '''
    passage
    phrase_text
    phrase_gloss
    head
    attributive
    #words_mother
    #nouns_mother
'''.strip().split()
nfields = len(fields)
row_template = ('{}\t' * (nfields - 1))+'{}\n'

of_path_template = 'attributives_{}.csv'
for fmt in ['ec', 'ha']:
    of = open(of_path_template.format(fmt), 'w')
    of.write('{}\n'.format('\t'.join(fields)))
    for s in sorted(attr_subphrase_mother, key=NK):
        sw = list(L.d('word', s))
        p = L.u('phrase', s)
        pw = list(L.d('word', p))
        m = attr_subphrase_mother[s]
        mw = list(L.d('word', m))

        of.write(row_template.format(
            T.passage(s),
            T.words(pw, fmt=fmt).replace('\n', ' '),
            ' '.join(F.gloss.v(w) for w in pw),
            T.words(mw, fmt=fmt).replace('\n', ' '),
            T.words(sw, fmt=fmt).replace('\n', ' '),
            len(mw),
            len([w for w in mw if F.sp.v(w) == 'subs']),
        ))

    of.close()
    inf('Written {} lines to {}'.format(len(attr_subphrase_mother) + 1, of_path_template.format(fmt)))

print(open(of_path_template.format('ec')).read()[0:1000])



