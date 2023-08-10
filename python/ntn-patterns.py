import sys
import collections
import subprocess

from lxml import etree

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
API = fabric.load('etcbc{}'.format(version), 'lexicon', 'ntn', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads
        function
        g_word_utf8 trailer_utf8
        lex prs sp nametype
        book chapter verse label number
    ''',''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

locative_lexemes = {
    '>RY/',
    'BJT/',
    'DRK/',
    'HR/',
    'JM/',
    'JRDN/',
    'JRWCLM/',
    'JFR>L/',
    'MDBR/',
    'MW<D/',
    'MZBX/',
    'MYRJM/',
    'MQWM/',
    'SBJB/',
    '<JR/',
    'FDH/',
    'CM',
    'CMJM/',
    'CMC/',
    'C<R/',
}
no_prs = {'absent', 'n/a'}

statclass = {
    'o': 'info',
    '+': 'good',
    '-': 'error',
    '?': 'warning',
    '!': 'special',
    '*': 'note',
}
statsym = dict((x[1], x[0]) for x in statclass.items())

def cert_status(cert):
    if cert == 0: return 'error'
    elif cert == 1: return 'warning'
    elif cert <= 10: return 'good'
    else: return 'special'

tsvfile = outfile('ntn.csv')
notefile = outfile('ntn-note.csv')
nresults = 0
nclauses = 0
orders = collections.Counter()
certs = collections.Counter()
tsvfile.write('book\tchapter\tverse\torder\tverb\tobject\tloc\tloc\tloc\tloc\tind\tind\tind\tcomplement text\tca_num\tclause text\n')
tsvfile.write('book\tchapter\tverse\torder\tverb\tobject\t# loc lexemes\t# topo\t# prep_b\tlocativity\t# prep_l\t# L prop\tindirect object\tcomplement text\tca_num\tclause text\n')
pclass = collections.Counter()
pclass['LI'] = 0
notefile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
    'version', 'book', 'chapter', 'verse', 'clause_atom', 'is_shared', 'is_published', 'status', 'keywords', 'ntext',
))
keywords = 'ntn-loca'
is_shared = 'T'
is_published = ''
status = statsym['info']
ntext_fmt = 'locative versus indirect object: L={} I={}; {}'

climit = 900

kws = ''.join(' {} '.format(k) for k in set(keywords.strip().split()))

for clause in F.otype.s('clause'):
    nclauses += 1
    phrases = {}
    order = ''
    verb = None
    for phrase in L.d('phrase', clause):
        pf = F.function.v(phrase)
        if pf in {'Pred', 'Objc', 'Cmpl'}:
            words = L.d('word', phrase)
            if pf not in phrases:
                order += pf[0]
                phrases[pf] = words
            else:
                phrases[pf].extend(words)
    is_ntn = False

    for w in phrases.get('Pred', []):
        if F.sp.v(w) == 'verb' and F.lex.v(w) == 'NTN[':
            is_ntn = True
            verb = w
            break
    if not is_ntn: continue
    nresults += 1    
    orders[order] += 1    

    book = F.book.v(L.u('book', verb))
    chapter = F.chapter.v(L.u('chapter', verb))
    verse = F.verse.v(L.u('verse', verb))    
    clause_atom = F.number.v(L.u('clause_atom', verb))
    
    verb_txt = F.g_word_utf8.v(verb)
    obj_txt = ''.join(F.g_word_utf8.v(x)+F.trailer_utf8.v(x) for x in phrases.get('Objc', []))
    cmpl_txt = ''.join(F.g_word_utf8.v(x)+F.trailer_utf8.v(x) for x in phrases.get('Cmpl', []))
    if len(cmpl_txt) > climit:
        cmpl_txt = cmpl_txt[0:climit]+'...'
    clause_txt = ''.join(F.g_word_utf8.v(x)+F.trailer_utf8.v(x) for x in L.d('word', clause))

    compl_wnodes = phrases.get('Cmpl', [])
    compl_lexemes = [F.lex.v(w) for w in compl_wnodes]
    compl_lset = set(compl_lexemes)
    lex_locativity = len(locative_lexemes & compl_lset)
    prep_b = len([x for x in compl_lexemes if x == 'B'])
    prep_l = len([x for x in compl_wnodes if F.lex.v(x) == 'L' and F.prs.v(x) not in no_prs])
    prep_lpr = 0
    lwn = len(compl_wnodes)
    for (n, wn) in enumerate(compl_wnodes):
        if F.lex.v(wn) == 'L':
            if n+1 < lwn:
                if F.sp.v(compl_wnodes[n+1]) == 'nmpr':
                    prep_lpr += 1
    topo = len([x for x in compl_wnodes if F.nametype.v(x) == 'topo'])

    loca = lex_locativity + topo + prep_b
    indi = prep_l + prep_lpr

    this_class = ''
    this_class += 'L' if loca else ''
    this_class += 'I' if indi else ''
    pclass[this_class] += 1
    
    tsvfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        book, 
        chapter, 
        verse,
        order,
        verb_txt,
        obj_txt,
        lex_locativity,
        topo,
        prep_b,
        loca,
        prep_l,
        prep_lpr,
        indi,
        ' '.join(compl_lexemes),
        clause_atom,
        clause_txt,
    ).replace('\n', ' ')+'\n')
    
    ntext = ntext_fmt.format(loca, indi, cmpl_txt)
    certainty = abs(loca - indi) * max((loca, indi))
    certs[certainty] += 1
    status = statsym[cert_status(certainty)]
    notefile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        version, book, chapter, verse, clause_atom, is_shared, is_published, status, kws, ntext,
    ).replace('\n', ' ')+'\n')
    
tsvfile.close()
notefile.close()
for order in sorted(orders):
    print("{:<5}: {:>3} results".format(order, orders[order]))

for cert in sorted(certs):
    print("{:>5} = {:<8}: {:>3} results".format(cert, cert_status(cert), certs[cert]))

for this_class in pclass:
    print("{:<2}: {:>3} results".format(this_class, pclass[this_class]))
print('Total: {:>3} results in {} clauses'.format(nresults, nclauses))

get_ipython().system("head -n 10 {my_file('ntn.csv')}")

get_ipython().system("head -n 10 {my_file('ntn-note.csv')}")



