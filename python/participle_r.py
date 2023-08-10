import sys, os
import collections
import re, csv

books = {
'AMOS': ('amos', 'amo'),
'CAN': ('song of songs', 'sol'),
'DAN': ('daniel', 'dan'),
'DEUT': ('deuteronomy', 'deu'),
'ESR': ('ezra', 'ezr'),
'EST': ('esther', 'est'),
'EXO': ('exodus', 'exo'),
'EZE': ('ezekiel', 'eze'),
'GEN': ('genesis', 'gen'),
'HAB': ('habakkuk', 'hab'),
'HAG': ('haggai', 'hag'),
'HOS': ('hosea', 'hos'),
'ICHR': ('1 chronicles', '1ch'),
'IICHR': ('2 chronicles', '2ch'),
'IIKON': ('2 kings', '2ki'),
'IISA': ('2 samuel', '2sa'),
'IKON': ('1 kings', '1ki'),
'IOB': ('job', 'job'),
'ISAM': ('1 samuel', '1sa'),
'JER': ('jeremiah', 'jer'),
'JES': ('isaiah', 'isa'),
'JOE': ('joel', 'joe'),
'JONA': ('jona', 'jon'),
'JOZ': ('joshua', 'jos'),
'LEV': ('leviticus', 'lev'),
'MAL': ('malachi', 'mal'),
'MICH': ('micah', 'mic'),
'NAH': ('nahum', 'nah'),
'NEH': ('nehemiah', 'neh'),
'NUM': ('numbers', 'num'),
'OBAD': ('obadiah', 'oba'),
'PRO': ('proverbs', 'pro'),
'PS': ('psalms', 'psa'),
'QOH': ('qoheleth', 'qoh'),
'RICHT': ('judges', 'jud'),
'RUTH': ('ruth', 'rut'),
'THR': ('lamentations', 'lam'),
'ZACH': ('zechariah', 'zec'),
'ZEP': ('zephaniah', 'zep'),
}
print("{} books".format(len(books)))

base_dir = '{}/Dropbox/laf-fabric-output/etcbc4b/participle'.format(os.path.expanduser('~'))
filepat = 'participia_compleet_r{}.csv'
start_column_names = {0: tuple('C{:>02d}'.format(c+1) for c in range(17))}
column_names = None
column_index = None

def infile(f): return open('{}/{}'.format(base_dir, f))
def outfile(f): return open('{}/{}'.format(base_dir, f), mode='w')
def msg(m):
    sys.stderr.write(m + '\n')
    sys.stderr.flush()

passage_pat = re.compile(r'([0-9]{2})\s*([A-Z_]+)\s*([0-9]+),([0-9]+)\.([0-9]+)')

errors = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
sourcef = None
targetf = None
data = None
new_data = None
nrows = None
levels = None

the_stage = 0

def error(cat, r, f):
    errors[the_stage][cat].append((r,f))
    
def make_passage(m):
    return (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))

def stage_start(nr=None):
    global the_stage
    global sourcef
    global targetf
    global column_names
    global column_index
    global data
    global new_data
    global nrows
    global levels
    
    if nr == None:
        the_stage += 1
    else:
        the_stage = nr
    msg("===BEGIN==STAGE {}=====".format(the_stage))
    column_names = list(start_column_names[the_stage - 1])
    column_index = dict((name, e) for (e, name) in enumerate(column_names))
    sfile = filepat.format(the_stage - 1)
    tfile = filepat.format(the_stage)
    msg('Column names before:\n{}'.format(', '.join(column_names)))
    msg('Reading participle text data stage {} ({} => {}) ...'.format(the_stage, sfile, tfile))
    errors[the_stage] = collections.defaultdict(lambda: [])
    sourcef = infile(sfile)
    targetf = outfile(tfile)
    data = csv.reader(sourcef)
    new_data = csv.writer(targetf)
    nrows = 0
    levels = collections.defaultdict(lambda: collections.Counter())


def stage_end(last=False):
    global targetf
    global columnindex
    
    start_column_names[the_stage] = tuple(column_names)
    column_index = dict((name, e) for (e, name) in enumerate(column_names))

    targetf.close()
    sourcef.close()
    tfile = filepat.format(the_stage)
    ffile = filepat.format('_final')
    targetf = infile(tfile)
    data = csv.reader(targetf)
    
    ncols = len(column_names)
    row_lengths = collections.Counter()

    if last:
        finalf = outfile(ffile)
        new_data = csv.writer(finalf)
        new_data.writerow(column_names)

    for row in data:
        row_lengths[len(row)] += 1
        for (e, field) in enumerate(row):
            levels[e][field] += 1
        if last:
            new_data.writerow(row)
    targetf.close()
    if last:
        finalf.close()

    show_n = 20
    for e in sorted(levels):
        valueset = levels[e].keys()
        lnv = len(valueset)
        examples = ' '.join(str(x) for x in sorted(valueset)[0:show_n])
        rest = ' ... {} more'.format(lnv - show_n) if lnv > show_n else ''
        print("{:<15} has {:>5} levels ({}{})".format(column_names[e], lnv, examples, rest))

    msg("{:>5} total rows x {:>3} header columns".format(nrows, ncols))
    for (rl, rw) in sorted(row_lengths.items(), key=lambda x: (-x[1], x[0])):
        msg("{:>5} body  rows x {:>3} actual columns ({})".format(rw, rl, 'OK' if rl == ncols else 'ERROR'))
    if errors:
        my_errors = errors[the_stage]
        for cat in sorted(my_errors):
            msg("Error: {} ({}x)".format(cat, len(my_errors[cat])))
            for (r, f) in my_errors[cat]:
                msg("\t{:>5}: {}".format(r, f))

    msg("===END====STAGE {}=====".format(the_stage))

def show_col(colname):
    global column_index
    column_index = dict((name, e) for (e, name) in enumerate(column_names))

    print("Start levels of column {}".format(colname))
    for (val, cnt) in sorted(levels[column_index[colname]].items(), key=lambda x: (-x[1], x[0])):
        print("{:<10}: {:>5}x".format(val, cnt))
    print("End levels of column {}".format(colname))

stage_start(nr=1)

rencols = dict(
    C02='state',
    C03='k',
    C04='domain',
    C06='phrf',
    C07='carc',
    C09='conj',
    C11='neg',
    C12='vstem',
    C14='lex',
    C16='clause',
    C17='comment',
)
delcols = '''C05 C17'''

for old in rencols:
    column_names[column_index[old]] = rencols[old]

delcols_sorted = sorted((column_index[x] for x in delcols.split()), reverse=True)

for dc in delcols_sorted:
    del column_names[dc:dc+1]

for row in data:
    for dc in delcols_sorted:
        del row[dc:dc+1]
    nrows += 1
    new_data.writerow(row)

stage_end()

stage_start(nr=2)

bookabbs = set()

typ1 = column_index['C01']
bookl = column_index['C15']
lex = column_index['lex']

for row in data:
    lexval = row[lex]
    if not lexval.endswith('['):
        error('lexeme not ending on [', nrows, lexval)
    row[lex] = lexval.strip('[')

    match = passage_pat.match(row[bookl])
    if match:
        (booknum, bookabb, chapter, verse, seqnum) = match.groups()
        bookabbs.add(bookabb)
        (book_name, book_acro) = books[bookabb]
        chapnum = int(chapter)
        versenum = int(verse)
        vlabel = '{}{}.{}'.format(book_acro, chapnum, versenum)
        row[bookl:bookl+1] = (
            vlabel,
            int(booknum), 
            book_name,
            book_acro,
            chapnum,
            versenum,
            int(seqnum),
        )
    else:
        error('Unrecognized passage', nrows, row[bookl])

    if row[typ1].count('-') > 2:
        error('More than 2 - in manual field', nrows, row[typ1])
    row[typ1:typ1+1] = (row[typ1].replace("'",'').replace('"','') + '-----').split('-')[0:3]

    row[typ1] = row[typ1].rstrip('"')
    while row[typ1].endswith('v'):
        row[typ1] = row[typ1][0:-1]
        row[typ1+2] += 'v'
    if row[typ1].startswith('n') and not (row[typ1] == 'n' or row[typ1][1] in {'w', '>'}):
        error('n followed by stray characters in manual field', nrows, '-'.join(row[typ1:typ1+2]))

    has_wav = 1 if 'w' in row[typ1] or 'w' in row[typ1+2] else 0
    has_alef = 1 if '>' in row[typ1] or '>' in row[typ1+2] else 0
    row[typ1:typ1+1] = [
        row[typ1].replace('w', '').replace('>', ''), 
        has_wav,
        has_alef,
    ]
    row[typ1+4] = row[typ1+4].replace('w', '').replace('>', '')
    if row[typ1] == 'N': row[typ1] = 'n'
    if row[typ1] == 'b2c': row[typ1] = 'b'
    if row[typ1].startswith('bijzin?nVv'):
        row[typ1] = 'n'
        row[typ1+3] = 'Vv'
    new_data.writerow(row)
    nrows += 1

column_names[bookl:bookl+1] = ('vlabel', 'booknum', 'bookname', 'bookacro', 'chapter', 'verse', 'seqnum')
column_names[typ1:typ1+1] = ('typ1', 'typ2', 'typ3')
column_names[typ1:typ1+1] = ('typ1', 't1_wav', 't1_alef')
            
stage_end()

stage_start(nr=3)

levels_0 = collections.defaultdict(lambda: collections.Counter())

typ1 = column_index['typ1']
typ2 = column_index['typ2']
typ3 = column_index['typ3']
dom = column_index['domain']
dlvs = ['D', 'N', 'Q', '?']

for row in data:
    d = row[dom]
    lend = len(d)
    these_dlvs = [0, 0, 0, 0]
    for (i, lv) in enumerate(dlvs):
        if d[-1] == lv:
            these_dlvs[i] = 1
        elif lend > 1 and d[-2] == lv:
            these_dlvs[i] = 0.5
    row[dom:dom+1] = these_dlvs + [lend]
    
    (f2, f3) = (row[typ2], 0)
    if row[typ1] in {'h', 'k'}:
        if f2.startswith('a'):
            f2 = f2.lstrip('a')
            f3 = 1
    row[typ2:typ2+1] = (f2, f3)
    
    nvs = row[typ3+1].count('v')
    row[typ3+1:typ3+2] = [
        row[typ3+1].replace('v',''), 
        nvs,
    ]
    new_data.writerow(row)
    if row[typ1] in {'p', 'h', 'k'}:
        levels_0[row[typ1]][row[typ2]] += 1
    nrows += 1

column_names[dom:dom+1] = ['dom_{}'.format(x) for x in dlvs] + ['dom_emb']
column_names[typ2:typ2+1] = ('typ2strip-a', 't2_a')
column_names[typ3+1:typ3+2] = ('typ3strip-v', 't3#v')

stage_end()

#show_col('typ1')
#show_col('typ3strip-v')

#for val0 in sorted(levels_0):
#    print("Levels of typ2strip-a if typ1 is {}:".format(val0))
#    for (val, occ) in sorted(levels_0[val0].items(), key=lambda x: (-x[1], x[0])):
#        print("\t{:<5} occurs {:>5}x".format(val, occ))    

stage_start(nr=4)

trans = '''
H   = HNH
Hs  = BLJ
>   = EJN
>s  = EJN
<   = OWD
J   = JC
P>  = EJN
<s  = OWD
PB  = BLJ
1>  = EJN
Js  = JC
r>  = EJN
r>s = EJN
><  = EJN,OWD
1>s = EJN
B   = BLJ
Pb> = EJN
hB  = BLJ
hJ  = JC
'''

trans_table = dict(
    (x.strip(),set(y.strip().split(','))) 
    for (x,y) in (
        z.strip().split('=') 
        for z in trans.split('\n') if z != ''
    )
)
t2_levels = set()
for x in trans_table: t2_levels |= trans_table[x]
t2_level_sorted = sorted(t2_levels)
ll = len(t2_levels)

typ1 = column_index['typ1']
typ2 = column_index['typ2strip-a']

for row in data:
    if row[typ1] == 'p':
        val = row[typ2]
        if val not in trans_table:
            error('Unrecognized level for typ2strip-a', nrows, val)
            row[typ2+1:typ2+1] = ['?' for x in t2_level_sorted]
        else:
            these_levels = trans_table[val]
            row[typ2+1:typ2+1] = [1 if x in these_levels else 0 for x in t2_level_sorted]
    else:
            row[typ2+1:typ2+1] = [0 for x in t2_level_sorted]
    nrows += 1
    new_data.writerow(row)

column_names[typ2+1:typ2+1] = ['t2_{}'.format(x) for x in t2_level_sorted]
stage_end()

#show_col('typ3strip-v')

stage_start(nr=5)

vstem = column_index['vstem']

for row in data:
    val = row[vstem]
    val2 = row[vstem+1]
    if val == 'qal' and val2 == 'pas': val += 'p'
    row[vstem] = val
    del row[vstem+1:vstem+2]
    nrows += 1
    new_data.writerow(row)

del column_names[vstem+1:vstem+2]
stage_end()
#show_col('vstem')

stage_start(nr=6)

conj = column_index['conj']

trans_conj = {
    '': 'empty',
    'W': 'w',
    'H': 'h',
    'KJ': 'ki',
    '>CR': 'acr',
    'W-/H': 'h',
    '>M': 'im',
    'DJ': 'di',
    'K->CR': 'kacer',
    'W-/>M': 'im',
    'C': 'c',
    'W-/KJ': 'ki',
    'L-H': 'h',
    '>T->CR': 'acr',
    'K-H': 'h',
    'PN': 'pn',
    'W-/W': 'w',
    '<D': 'ad',
    'K-L-QBL/-DJ': 'di',
    '<L-H': 'h',
    '>W': 'ow',
    'KJ->M': 'im',
    '>XR/->CR': 'acr',
    'KJ-/>M': 'im',
    'LW': None,
    'W-/>T-H': 'h',
    'J<N/': None,
    'K-DJ': 'di',
    'L->CR': 'acr',
    'LM<N': None,
    'MN-H': 'h',
    'W-/>CR': 'acr',
    '<D->CR': 'acr',
    '<D-C': 'c',
    '<D-H': 'h',
    '<L->CR': 'acr',
    '<L-DJ': 'di',
    '<L-KJ': 'ki',
    '>CR-/W': 'w',
    '>L-H': 'h',
    '>T-H': 'h',
    'B-C': 'c',
    'B-H': 'h',
    'B-VRM/': None,
    'H-/W': 'w',
        'J<N/->CR': 'acr',
    'K-C': 'c',
    'K-PH/->CR': 'acr',
    'KJ-/LWL>': None,
    'KMW': None,
    'LWL>': None,
    'MN': None,
    'MN-DJ': 'di',
    'MN-L-BD/-H': 'h',
    'TXT/->CR': 'acr',
    'W-/<L-H': 'h',
    'W-/>L-H': 'h ',
    'W-/B-KL/-DJ': 'di',
    'W-/DJ': 'di',
    'W-/L->CR': 'acr',
    'W-/L-H': 'h',
    'W-/LW': None,
    'W-/W-/W': 'w',
}

for row in data:
    val = row[conj]
    val2 = row[conj+1]
    if val2 != '': val = val2
    row[conj] = trans_conj[val] or ''
    del row[conj+1:conj+2]
    nrows += 1
    new_data.writerow(row)

del column_names[conj+1:conj+2]
stage_end()

stage_start(nr=7)

carc = column_index['carc']

for row in data:
    carc1 = ''
    carc2 = ''
    carc3 = ''
    code = str(row[carc])
    if code == '':
        error('Empty carc', nrows, code)
    if len(code) == 3 and code[0] == '2' and code[1:] not in {'00', '01'}:
        error('Strange carc in 200 range', nrows, code)
    if len(code) == 3 and code[0] == '2':
        code = str(row[carc+1])
    if code == '':
        carc1 = 'chain'
    elif code == '0':
        carc1 = 'txto'
    elif 10 <= int(code) <= 16:
        carc1 = 'rela'
        carc2 = code[1]
    elif 50 <= int(code) <= 74:
        carc1 = 'infc'
    elif len(code) == 2:
        error('Strange carc with two digits', nrows, code)
    elif code == '999':
        carc1 = 'q'
    else:
        (carc1,carc2,carc3) = (code[0], code[1], code[2])
    row[carc:carc+2] = (carc1, carc2, carc3)
    nrows += 1
    new_data.writerow(row)

column_names[carc:carc+2] = ('carc1', 'carc2', 'carc3')
stage_end()

stage_start(nr=8)

features = collections.OrderedDict((
    ('vstem', {
        'lvs': list(sorted('''qal qalp hif nif piel peal pual hit hof haf pael htpa hsht htpe pasq tif shaf'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('neg', {
        'lvs': list(sorted('''>JN/ >L= BLJ/ L> MN->JN/'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('conj', {
        'lvs': list(sorted('''acr ad c di empty h h  im kacer ki ow pn w'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('carc3', {
        'lvs': list(sorted('''0 1 2 3 4 5 6 7'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('carc2', {
        'lvs': list(sorted('''0 1 2 3 4 5 6 7'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('carc1', {
        'lvs': list(sorted('''1 3 4 5 6 7 8 chain infc q rela txto'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('phrf', {
        'lvs': list(sorted('''AdjP AdvP DPrP NP PP PPrP PrNP VP'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('k', {
        'lvs': list(sorted('''#NAAM? +K +K='''.split(), reverse=True)), 
        'keep': False,
    }),
    ('state', {
        'lvs': list(sorted(''': :a :c :e'''.split(), reverse=True)), 
        'keep': False,
    }),
    ('typ3strip-v', {
        'lvs': list(sorted('''V PC S O E D Nl Nhl'''.split())),
        'keep': True,
        'lvname': 't3',
    }),
))

colnums = []
clevels = []
keep = []
for feat in (features):
    colnums.append(column_index[feat])
    clevels.append(features[feat]['lvs'])
    keep.append(features[feat]['keep'])

for row in data:
    for (i, cn) in enumerate(colnums):
        val = row[cn]
        flags = []
        for lv in clevels[i]:
            if lv in val:
                if keep[i]: val = val.replace(lv, '')
                flag = 1
            else:
                flag = 0
            flags.append(flag)
        row[cn:cn+1] =  ([val] if keep[i] else []) + flags
    new_data.writerow(row)
    nrows += 1
    
for feat in features:
    cn = column_index[feat]
    keep = features[feat]['keep']
    lvname = features[feat].get('lvname', feat)
    lvs = features[feat]['lvs']
    column_names[cn:cn+1] = ([feat] if keep else []) + ['{}_{}'.format(lvname, x) for x in lvs]

stage_end(last=True)

#for feat in features:
#    if features[feat]['keep']: show_col(feat)



