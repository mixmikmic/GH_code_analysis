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
        function lex
        gloss
    ''',
    '''
    '''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

occurrences = collections.defaultdict(lambda: set())
# a defaultdict is needed for the case where we see a lexeme for the first time.
# In that case occurrences[lexeme] does not yet exist.
# The defaultdict then inserts the key lexeme with value empty set into the dict.
inf('Making occurrence index ...')
for w in F.otype.s('word'):
    occurrences[F.lex.v(w)].add(w)
inf('{} lexemes'.format(len(occurrences)))

def bits(fmt, w):
    p = L.u('phrase', w)
    pw = list(L.d('word', p))
    return (
        T.passage(w),
        p,
        T.words(pw, fmt=fmt).replace('\n', ' '),
        ' '.join(F.gloss.v(x) for x in pw),
        F.function.v(p),
        F.lex.v(w),
        T.words([w], fmt=fmt).replace('\n', ' '),
        w,
    )

fields = '''
    passage
    phrase_node
    phrase_text
    phrase_gloss
    phrase_function
    lexeme
    occ_text
    occ_node
'''.strip().split()
nfields = len(fields)
row_template = ('{}\t' * (nfields - 1))+'{}\n'
of_path_template = 'occurrences_{}.{}.csv'

def lex_file_name(lexeme):
    # in order to use the lexeme in a file name, we replace < > / [ = by harmless characters
    return lexeme.        replace('/', 's').        replace('[', 'v').        replace('=', 'x').        replace('<', 'o').        replace('>', 'a')

def lex_info(lexeme, fmt):
    file_lex = lex_file_name(lexeme)
    file_name = of_path_template.format(file_lex, fmt)
    of = open(file_name, 'w')
    of.write('{}\n'.format('\t'.join(fields)))
    if lexeme not in occurrences:
        msg('There is no lexeme "{}"'.format(lexeme))
        occs = []
    else:
        occs = sorted(occurrences[lexeme], key=NK)
        # sorted turns a set into a list. The order is given by the key parameter.
        # This is the function NK (see the ETCBC-reference. It orders nodes
        # according to where their associated text occurs in the Bible
    for w in occs:
        of.write(row_template.format(*bits(fmt, w)))
        # bits yields a tuple of values. The * unpacks this tuple in separate arguments.
    of.close()
    inf('Written {} lines to {}'.format(len(occs) + 1, file_name))

def show_stats(lexeme):
    # we produce an overview of the distribution of the occurrences over the books
    # book names in Swahili
    book_dist = collections.Counter()
    if lexeme not in occurrences:
        msg('There is no lexeme "{}"'.format(lexeme))
        occs = []
    else:
        occs = sorted(occurrences[lexeme], key=NK)
    for w in occs:
        book_node = L.u('book', w)
        book_name_sw = T.book_name(book_node, lang='sw')
        book_name = T.book_name(book_node)
        book_dist['{:<30} = {}'.format(book_name_sw, book_name)] += 1
    # we sort the results by frequency
    total = 0
    for (b, n) in sorted(book_dist.items(), key=lambda x: (-x[1], x[0])):
        print('{:<10} has {:>5} occurrences in {}'.format(lexeme, n, b))
        total += n
    print('{:<10} has {:>5} occurrences in {}'.format(lexeme, total, 'the whole Bible'))

lexeme = '>LHJM/'
show_stats(lexeme)
lex_info(lexeme, 'ec')
lex_info(lexeme, 'ha')

print(open(of_path_template.format(lex_file_name(lexeme), 'ec')).read()[0:1000])



