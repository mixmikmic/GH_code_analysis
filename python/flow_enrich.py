import sys,os
import collections

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.extra import ExtraData

fabric = LafFabric()

source = 'etcbc'
version = '4b'

API = fabric.load(source+version, '--', 'flow_corr', {
    "xmlids": {"node": True, "edge": False},
    "features": ('''
        oid otype
        sp vs lex
        function
        chapter verse
    ''',''),
    "prepare": prepare,
}, verbose='NORMAL')
exec(fabric.localnames.format(var='fabric'))

verbs = set('''
    CJT
    BR>
    QR>
'''.strip().split())

msg('Finding occurrences')
occs = collections.defaultdict(list)
for n in F.otype.s('word'):
    lex = F.lex.v(n)
    if lex.endswith('['):
        lex = lex[0:-1]
        occs[lex].append(n)
msg('Done')
for verb in sorted(verbs):
    print('{} {:<5} occurrences'.format(verb, len(occs[verb])))

COMMON_FIELDS = '''
    cnode#
    vnode#
    pnode#
    book
    chapter
    verse
    verb_lexeme
    verb_occurrence
'''.strip().split()

CLAUSE_FIELDS = '''
    clause_text    
'''.strip().split()

PHRASE_FIELDS = '''
    phrase_text
    function
    function_(corr)
    valence
    lexical
    grammatical
    semantical
'''.strip().split()

field_names = []
for f in COMMON_FIELDS: field_names.append(f)
for i in range(max((len(CLAUSE_FIELDS), len(PHRASE_FIELDS)))):
    pf = PHRASE_FIELDS[i] if i < len(PHRASE_FIELDS) else '--'
    field_names.append(pf)
    
fillrows = len(CLAUSE_FIELDS) - len(PHRASE_FIELDS)
cfillrows = 0 if fillrows >= 0 else -fillrows
pfillrows = fillrows if fillrows >= 0 else 0
print('\n'.join(field_names))    

def vfile(verb, kind): return '{}_{}_{}{}.csv'.format(verb.replace('>','a').replace('<', 'o'), kind, source, version)

def gen_sheet(verb):
    rows = []
    fieldsep = ';'
    for wn in occs[verb]:
        cn = L.u('clause', wn)
        vn = L.u('verse', wn)
        bn = L.u('book', wn)
        book = T.book_name(bn, lang='en')
        chapter = F.chapter.v(vn)
        verse = F.verse.v(vn)
        vl = F.lex.v(wn).rstrip('[=')
        vt = T.words([wn], fmt='ec').replace('\n', '')
        ct = T.words(L.d('word', cn), fmt='ec').replace('\n', '')
        
        common_fields = (cn, wn, -1, book, chapter, verse, vl, vt)
        clause_fields = (ct,)
        rows.append(common_fields + clause_fields + (('',)*cfillrows))
        for pn in L.d('phrase', cn):
            common_fields = (cn, wn, pn, book, chapter, verse, vl, vt)
            pt = T.words(L.d('word', pn), fmt='ec').replace('\n', '')
            pf = F.function.v(pn)
            phrase_fields = (pt, pf) + (('',)*5)
            rows.append(common_fields + phrase_fields + (('',)*pfillrows))
    filename = vfile(verb, 'blank')
    row_file = outfile(filename)
    row_file.write('{}\n'.format(fieldsep.join(field_names)))
    for row in rows:
        row_file.write('{}\n'.format(fieldsep.join(str(x) for x in row)))
    row_file.close()
    msg('Generated correction sheet for verb {}'.format(filename))
    
for verb in verbs: gen_sheet(verb)    

def read_corr(rootdir):
    results = []
    for verb in verbs:
        filename = '{}/{}'.format(rootdir, vfile(verb, 'corrected'))
        if not os.path.exists(filename):
            print('NO file {}'.format(filename))
            continue
        with open(filename) as f:
            header = f.__next__()
            for line in f:
                fields = line.rstrip().split(';')
                for i in range(1, len(fields)//4):
                    (pn, pc) = (fields[4*i], fields[4*i+3])
                    pc = pc.strip()
                    if pc != '': results.append((int(pn), pc))
        print('{}: Found {:>5} corrections in {}'.format(verb, len(results), filename))
    return results

corr = ExtraData(API)
corr.deliver_annots(
    'complements', 
    {'title': 'Verb complement corrections', 'date': '2016-02'},
    [
        ('cpl', 'complements', read_corr, (
            ('JanetDyk', 'ft', 'function'),
        ))
    ],
)

API=fabric.load(source+version, 'complements', 'flow_corr', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype
        JanetDyk:ft.function etcbc4:ft.function
    ''',
    '''
    '''),
    "prepare": prepare,
}, verbose='NORMAL')
exec(fabric.localnames.format(var='fabric'))

msg('Checking corrections')
corr = collections.Counter()
errors = collections.Counter()
corrected_nodes = set()
for n in NN():
    c = F.JanetDyk_ft_function.v(n)
    if c != None:
        if F.otype.v(n) != 'phrase': errors['Correction applied to non-phrase object'] += 1
        if n in corrected_nodes: errors['Phrase with multiple corrections'] += 1
        o = F.etcbc4_ft_function.v(n) or ''
        corr[(o, c)] += 1
        corrected_nodes.add(n)
    
msg('Found {} types of corrections'.format(len(corr)))
print(corr)
for ((o, c), n) in sorted(corr.items(), key=lambda x: (-x[1], x[0])):
    print('{:<5} => {:<5} {:>5} x'.format(o, c, n))
if not errors:
    print('NO ERRORS DETECTED')
else:
    print('THERE ARE ERRORS:')
    for (e, n) in sorted(errors.items(), key=lambda x: (-x[1], x[0])):
        print('{:>5} x {}'.format(n, e))

for p in F.otype.s('phrase'):
    if F.function.v(p) != 'EPPr': continue
    words = L.d('word', p)
    first_word = words[0]
    b = L.u('book', first_word)
    v = L.u('verse', first_word)
    c = L.u('clause', first_word)
    passage = '{} {}:{}'.format(
        T.book_name(b, lang='en'), 
        F.chapter.v(v),
        F.verse.v(v),
    )
    pt = T.words(L.d('word', p), fmt='ec').replace('\n', '')
    ct = T.words(L.d('word', c), fmt='ec').replace('\n', '')
    print('{} {} :: {}'.format(passage, pt, ct))



