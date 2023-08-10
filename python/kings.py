import sys,os, re, pickle
import collections, difflib
from IPython.display import HTML, display_pretty, display_html
from difflib import SequenceMatcher
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.lib import Transcription
fabric = LafFabric()

source = 'etcbc'
version = '4b'

API = fabric.load(source+version, 'lexicon', 'kings', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype
        language lex lex_utf8 g_cons_utf8 g_word_utf8 trailer_utf8 phono phono_sep
        book chapter verse label number
    ''',''),
    "prepare": prepare,
    "primary": False,
}, verbose='NORMAL')
exec(fabric.localnames.format(var='fabric'))

REFBOOKS = {'Reges_II'}
REFCHAPTERS = set(range(19,26))

Q1_FILE = 'qisaa_an.txt'
SHEBANQ_PATH = os.path.abspath('{}/../../../shebanq'.format(os.getcwd))
#CROSSREF_DB = '{}/static/docs/tools/parallel/files/crossrefs_lcs_db.txt'.format(SHEBANQ_PATH)
CROSSREF_DB = '{}/static/docs/tools/parallel/files/crossrefdb.csv'.format(SHEBANQ_PATH)
#PASSAGE_FMT = '{}.{}.{}'
PASSAGE_FMT = '{}~{}:{}'

SIM_THRESHOLD = 79 # the smallest value that causes Leviticus to be left out

NCOL_FILE = 'these_crossrefs.ncol'
ALL_VERSES_FILE = 'all_verses.txt'
NEW_MATRIX_FILE = 'new_matrix.tsv'

book_node = dict()
for b in F.otype.s('book'):
    book_name = F.book.v(b)
    book_node[book_name] = b
    if book_name == 'Reges_II':
        book_node[book_name+'r'] = b

trans_final_pat = re.compile('([KMNPY])(?= |\Z)')

def trans_final_repl(match): return match.group(1).lower()

msg('reading Q1')
qf = open(Q1_FILE)
q1 = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
nwords = 0
for line in qf:
    nwords += 1
    (passage, word, xword) = line.strip().split()
    (chapter, verse) = passage.split(',')
    word = trans_final_pat.sub(trans_final_repl, word)
    q1[int(chapter)][int(verse)].append(Transcription.to_hebrew_x(word))
qf.close()
msg('{} words in {} chapters in {} verses'.format(nwords, len(q1), sum(len(q1[x]) for x in q1)))

print(' '.join(q1[1][1]))

cliques = collections.defaultdict(lambda: set())
msg('Reading crossrefs database and picking out the relevant ones')
n = 0
all_verses = set()
new_sim_matrix = {}
ncoldata = []
with open(CROSSREF_DB) as f:
    for line in f:
        n += 1
        if n == 1: continue
        (bkx, chx, vsx, bky, chy, vsy, rd) = line.rstrip('\n').split('\t')
        if int(rd) < SIM_THRESHOLD: continue
        (chx, vsx, chy, vsy) = (int(z) for z in (chx, vsx, chy, vsy))
        if bkx in REFBOOKS and chx in REFCHAPTERS: cliques[(bkx, chx, vsx)].add((bky, chy, vsy))
        if bky in REFBOOKS and chy in REFCHAPTERS: cliques[(bky, chy, vsy)].add((bkx, chx, vsx))
        if (bkx in REFBOOKS and chx in REFCHAPTERS) or bky in REFBOOKS and chy in REFCHAPTERS:
            if bkx in REFBOOKS and chx in REFCHAPTERS: bkx += 'r'
            if bky in REFBOOKS and chy in REFCHAPTERS: bky += 'r'
            all_verses.add((bkx, chx, vsx))
            all_verses.add((bky, chy, vsy))
            if bkx == 'Reges_IIr':
                new_sim_matrix.setdefault(PASSAGE_FMT.format(bkx, chx, vsx), {})[PASSAGE_FMT.format(bky, chy, vsy)] = int(rd)
                ncoldata.append(((bkx, chx, vsx), (bky, chy, vsy), rd))
            else:
                new_sim_matrix.setdefault(PASSAGE_FMT.format(bky, chy, vsy), {})[PASSAGE_FMT.format(bkx, chx, vsx)] = int(rd)
                ncoldata.append(((bky, chy, vsy), (bkx, chx, vsx), rd))
all_verses_srt = sorted(all_verses, key=lambda x: (book_node[x[0]], x[1], x[2]))
    
ncolfile = open(NCOL_FILE, 'w')
for (x, y, r) in sorted(ncoldata, key=lambda z: (
        book_node[z[0][0]], z[0][1], z[0][2], 
        book_node[z[1][0]], z[1][1], z[1][2],
)):
    ncolfile.write('{} {} {}\n'.format(PASSAGE_FMT.format(*x), PASSAGE_FMT.format(*y), r))
ncolfile.close()
    
msg('{} entries read'.format(n))
countrefs = collections.Counter()
for x in cliques: countrefs[len(cliques[x])+1] += 1
for x in sorted(countrefs): msg('Cliques of length {}: {}'.format(x, countrefs[x]))
all_books = {x[0] for x in all_verses}
all_chapters = {(x[0], x[1]) for x in all_verses}
msg('{}, {}, {} relevant verses, chapters, books'.format(len(all_verses), len(all_chapters), len(all_books)))
print(' '.join(sorted(all_books)))

mfile = open(NEW_MATRIX_FILE, 'w')
headrow = '\t'.join(PASSAGE_FMT.format(*x) for x in all_verses_srt)
mfile.write('passage\t{}\n'.format(headrow))
for x in all_verses_srt:
    row = '\t'.join(str(new_sim_matrix.get(PASSAGE_FMT.format(*x), {}).get(PASSAGE_FMT.format(*y), 0)) for y in all_verses_srt)
    mfile.write('{}\t{}\n'.format(PASSAGE_FMT.format(*x), row))
mfile.close()

allvfile = open(ALL_VERSES_FILE, 'w')
for p in sorted(all_verses, key=lambda x: (book_node[x[0]], x[1], x[2])):
    allvfile.write('{}~{}:{}\n'.format(*p))
allvfile.close()

g = nx.read_weighted_edgelist(NCOL_FILE)

gcolors = dict(
    Haggai=(0.7, 0.7, 0.7),
    Reges_IIr=(0.9, 0.9, 0.9),
    Reges_II=(1.0, 1.0, 0.3),
    Reges_I=(0.3, 0.3, 1.0),
    Chronica_II=(1.0, 0.3, 1.0),
    Jeremia=(0.3, 1.0, 0.3),
    Jesaia=(1.0, 0.3, 0.3),
)

all_books_cust = '''
    Haggai Jeremia Jesaia Reges_I Reges_II Reges_IIr Chronica_II
'''.strip().split()

offset_y = dict(
    Haggai=55,
    Reges_IIr=0,
    Reges_II=65,
    Reges_I=65,
    Chronica_II=50,
    Jeremia=70,
    Jesaia=0,
    Leviticus=0,
    Genesis=0,
)

ncolors = [gcolors[x.split('~')[0]] for x in g.nodes()]
nlabels = dict((x, x.split('~')[1]) for x in g.nodes())
ncols = len(all_books)
pos_x = dict((x, i) for (i,x) in enumerate(all_books_cust))
verse_lists = collections.defaultdict(lambda: [])
for (bk, ch, vs) in sorted(all_verses):
    verse_lists[bk].append('{}:{}'.format(ch, vs))
nrows = max(len(verse_lists[bk]) for bk in all_books_cust)
pos = {}
for bk in verse_lists:
    for (i, chvs) in enumerate(verse_lists[bk]):
        pos['{}~{}'.format(bk, chvs)] = (pos_x[bk], i+offset_y[bk])

plt.figure(figsize=(12,64))

nx.draw_networkx(g, pos,
    width=[g.get_edge_data(*x)['weight']/40 for x in g.edges()],
    edge_color=[g.get_edge_data(*x)['weight']/1.2 for x in g.edges()],
    edge_cmap=plt.cm.Greys,
    edge_vmin=50,
    edge_vmax=100,
    node_color=ncolors,
    node_size=100,
    labels=nlabels,
    alpha=0.4,
    linewidths=0,
)
#plt.axis('tight')
plt.ylim(-2, 121)
book_font_size = 14
plt.grid(b=True, which='both', axis='x')
plt.title('Parallels involving Reges_II 19-25', fontsize=24)
plt.text(1,70, '''
The parallels with
Reges_I and the other
chapters of Reges_II
are weaker and 
more sporadic.
Note that there are
also links within
Reges_II 19-25.
All these are probably
similar verses but
not parallels.
''', #bbox=dict(width=145, height=200, facecolor='yellow', alpha=0.4), fontsize=12)
     # suddenly the width and height keyword args are no longer accepted.
     # bbox performs an auto fit
     bbox=dict(facecolor='yellow', alpha=0.4), fontsize=12)
for bk in all_books_cust:
    for (ypos, suppress) in ((-4, False), (49.5, True), (83, True), (120, False)):
        if not suppress or bk != 'Reges_IIr':
            plt.text(pos_x[bk], ypos, bk, fontsize=book_font_size, horizontalalignment='center')

plt.savefig('kings_parallels.pdf')

corresp_spec = (
    (('Reges_II', (19, 1), (20,19)), ('Jesaia',      (37, 1), (39, 8))),
    (('Reges_II', (21, 1), (23, 3)), ('Chronica_II', (33, 1), (34,31))),
    (('Reges_II', (23,31), (25,30)), ('Jeremia',     (52, 1), (52,34))),
)

msg("Making a mapping between a passage specification and a verse node")
# we also make a list of all verses in all reference chapters
verses = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
passage2vnode = {}
for vn in F.otype.s('verse'):
    bk = F.book.v(vn)
    ch = int(F.chapter.v(vn))
    vs = int(F.verse.v(vn))
    lab = (bk, ch, vs)
    passage2vnode[lab] = vn
    verses[bk][ch].append(vs)
msg("{} verses".format(len(passage2vnode)))

def get_verselist(bk, start, end):
    (start_ch, start_vs) = start
    (end_ch, end_vs) = end
    result_list = []
    for ch in range(start_ch, end_ch+1):
        lower_vs = start_vs if ch == start_ch else 1
        upper_vs = end_vs if ch == end_ch else verses[bk][ch][-1]
        for vs in range(lower_vs, upper_vs+1):
            result_list.append((bk, ch, vs))
    return tuple(result_list)

compare_passages_proto = []
refverses = set()
parverses = set()
for (ref, par) in corresp_spec:
    reflist = get_verselist(*ref)
    parlist = get_verselist(*par)
    refverses |= set(reflist)
    parverses |= set(parlist)
    compare_passages_proto.append((
        reflist,
        parlist,
    ))

compare_passages_proto[0]

cliques_reduced = collections.defaultdict(lambda: set())
for ref in set(cliques.keys()) & refverses:
    pars = cliques[ref] & parverses
    cliques_reduced[ref] = pars
    for par in pars: cliques_reduced[par].add(ref)

compare_passages_multi = []
for (ref, par) in compare_passages_proto:
    compare_passages_multi.append((
        tuple((x, cliques_reduced[x]) for x in ref),
        tuple((x, cliques_reduced[x]) for x in par),
    ))

compare_passages_multi[0]

no_verse = ('', 0, 0)

def singlify(vmulti):
    vsingle = []
    for (x, xset) in vmulti:
        if len(xset) == 0:
            vsingle.append((x, no_verse))
        else:
            for y in sorted(xset):
                vsingle.append((x, y))
    return tuple(vsingle)

compare_passages = []
for (ref, par) in compare_passages_multi:
    compare_passages.append((singlify(ref), singlify(par)))

compare_passages[0]

normalized_passages = []
for (ref, par) in compare_passages:
    result = list(ref)
    insert_par = sorted(x for x in par if x[1] == no_verse)
    all_par = {x[0] for x in par}

    seen = [] # seen[i] contains the set of all vops seen in ref[0:i+1] 
    for (vr, vop) in result:
        seen.append((seen[-1] if len(seen) else set()) | (set() if vop == no_verse else {vop} ))
    
    for p in insert_par:
        vp = p[0]

        # determine for every position in result the smallest vp which has not been seen so far
        smallest_not_seen = [min(all_par - seeni) for seeni in seen]
        
        # the insert point for p is the smallest position where all passages before vp have been seen
        # if vp == smallest_not_seen[i] for a certain i, then vp has not been seen, but all vps smaller
        # than vp have been seen. The first i where this occurs, is the index we are after
        insert_point = min(i for i in range(0, len(smallest_not_seen)) if vp == smallest_not_seen[i]) + 1
        
        # now insert vp at the newly found instertion point.
        result.insert(insert_point, (no_verse, vp))
        
        # we also have to update the seen list
        seen = [] # seen[i] contains the set of all vops seen in ref[0:i+1] 
        for (vr, vop) in result:
            seen.append((seen[-1] if len(seen) else set()) | (set() if vop == no_verse else {vop} ))

    normalized_passages.append(tuple(result))

normalized_passages

def lex_diff(vr, vp):
    vnr = passage2vnode[vr] if vr[0] else None
    vnp = passage2vnode[vp] if vp[0] else None
    if vnr == None or vnp == None:
        return (set(), set())
    lexr = {F.lex_utf8.v(w).rstrip('/[=') for w in L.d('word', vnr)}
    lexp = {F.lex_utf8.v(w).rstrip('/[=') for w in L.d('word', vnp)}
    return (lexr-lexp, lexp-lexr)

compare_lexemes = []
for passage in normalized_passages:
    compare_passage = []
    for (r, p) in passage: compare_passage.append(lex_diff(r,p))
    compare_lexemes.append(compare_passage)

compare_lexemes[0]

own_lexemes = [] 

for (i, (r, p)) in enumerate(corresp_spec):
    my_own_lexemes = collections.defaultdict(lambda: collections.Counter())
    bookr = r[0]
    bookp = p[0]
    for (setr, setp) in compare_lexemes[i]:
        for x in setr: my_own_lexemes[bookr][x] += 1
        for x in setp: my_own_lexemes[bookp][x] += 1
    own_lexemes.append(my_own_lexemes)

own_lexemes[0]

css = '''
<style type="text/css">
table.t {
    width: 100%;
    border-collapse: collapse;
}
table.h {
    direction: rtl;
}
table.p {
    direction: ltr;
}
tr.t.tb {
    border-top: 2px solid #aaaaaa;
    border-left: 2px solid #aaaaaa;
    border-right: 2px solid #aaaaaa;
}
tr.t.bb {
    border-bottom: 2px solid #aaaaaa;
    border-left: 2px solid #aaaaaa;
    border-right: 2px solid #aaaaaa;
}
th.t {
    font-family: Verdana, Arial, sans-serif;
    font-size: large;
    vertical-align: middle;
    text-align: center;
    padding-left: 2em;
    padding-right: 2em;
    padding-top: 1ex;
    padding-bottom: 2ex;
    border-left: 2px solid #aaaaaa;
    border-right: 2px solid #aaaaaa;
}
td.t {
    border-left: 2px solid #aaaaaa;
    border-right: 2px solid #aaaaaa;
    padding-left: 1em;
    padding-right: 1em;
    padding-top: 0.3ex;
    padding-bottom: 0.5ex;
}
td.h {
    font-family: Ezra SIL, SBL Hebrew, Verdana, sans-serif;
    font-size: x-large;
    line-height: 1.7;
    text-align: right;
    direction: rtl;
}
td.ld {
    font-family: Ezra SIL, SBL Hebrew, Verdana, sans-serif;
    font-size: medium;
    line-height: 1.2;
    text-align: right;
    vertical-align: top;
    direction: rtl;
    width: 10%;
}
td.p {
    font-family: Verdana, sans-serif;
    font-size: large;
    line-height: 1.3;
    text-align: left;
    direction: ltr;
}
td.vl {
    font-family: Verdana, Arial, sans-serif;
    font-size: small;
    text-align: right;
    vertical-align: top;
    color: #aaaaaa;
    width: 5%;
    direction: ltr;
    border-left: 2px solid #aaaaaa;
    border-right: 2px solid #aaaaaa;
    padding-left: 0.4em;
    padding-right: 0.4em;
    padding-top: 0.3ex;
    padding-bottom: 0.5ex;
}
span.m {
    background-color: #aaaaff;
}
span.f {
    background-color: #ffaaaa;
}
span.x {
    background-color: #ffffaa;
    color: #bb0000;
}
span.delete {
    background-color: #ffaaaa;
}
span.insert {
    background-color: #aaffaa;
}
span.replace {
    background-color: #ffff00;
}
</style>
'''

diffhead = '''
<head>
    <meta http-equiv="Content-Type"
          content="text/html; charset=UTF-8" />
    <title></title>
    <style type="text/css">
        table.diff {
            font-family: Ezra SIL, SBL Hebrew, Verdana, sans-serif; 
            font-size: x-large;
            text-align: right;
        }
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    </style>
</head>
'''

html_file_tpl = '''<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<title>{}</title>
{}
</head>
<body>
{}
</body>
</html>'''

def print_label(vl, without_book=True):
    bookrep = '' if without_book else '{} '.format(vl[0])
    return '{}{}:{}'.format(bookrep, vl[1], vl[2]) if vl[0] != '' else ''

def print_diff(a, b):
    arep = ''
    brep = ''
    for (lb, ai, aj, bi, bj) in SequenceMatcher(isjunk=None, a=a, b=b, autojunk=False).get_opcodes():
        if lb == 'equal':
            arep += a[ai:aj]
            brep += b[bi:bj]
        elif lb == 'delete':
            arep += '<span class="{}">{}</span>'.format(lb, a[ai:aj])
        elif lb == 'insert':
            brep += '<span class="{}">{}</span>'.format(lb, b[bi:bj])
        else:
            arep += '<span class="{}">{}</span>'.format(lb, a[ai:aj])
            brep += '<span class="{}">{}</span>'.format(lb, b[bi:bj])
    return (arep, brep)
    
def get_vtext(v, hp):
    if hp == 'h':
        return ''.join('{}{}'.format(
            F.g_word_utf8.v(w), F.trailer_utf8.v(w)) for w in L.d('word', v),
        )
    if hp == 'p':
        return ''.join('{}{}'.format(
            F.phono.v(w), F.phono_sep.v(w)) for w in L.d('word', v),
        )
    return ''

def print_chunk(vr, vp, hp):
    vnr = passage2vnode[vr] if vr[0] else None
    vnp = passage2vnode[vp] if vp[0] else None
    textr = get_vtext(vnr, hp) if vnr != None else ''
    textp = get_vtext(vnp, hp) if vnp != None else ''
    (lexdiff_r, lexdiff_p) = lex_diff(vr, vp)
    (liner, linep) = print_diff(textr, textp)
    return '''
<tr class="t tb">
    <td class="vl">{br}</td>
    <td class="t {hp}">{lr}</td>
    <td class="t ld"><span class="delete">{ldr}</span></td>
    <td class="t ld"><span class="insert">{ldp}</span></td>
    <td class="t {hp}">{lp}</td>
    <td class="vl">{bp}</td>
</tr>
'''.format(
    br=print_label(vr),
    lr=liner,
    ldr=' '.join(sorted(lexdiff_r)),
    ldp=' '.join(sorted(lexdiff_p)),
    bp=print_label(vp), 
    lp=linep,
    hp=hp,
)

def print_passage(cmp_list, hp):
    result = []
    for item in cmp_list:
        result.append(print_chunk(item[0], item[1], hp))
    return '\n'.join(result)

def get_lex_summ(book, my_own_lex):
    result = []
    for (lex, n) in sorted(my_own_lex[book].items(), key=lambda x: (-x[1], x[0])):
        result.append('<span class="ld">{}</span>&nbsp;{}<br/>'.format(lex, n))
    return '\n'.join(result)
    
def print_lexeme_summary(bookr, bookp, my_own_lex):
    return '''
<tr class="t tb">
    <td class="vl">&nbsp;</td>
    <td class="t">&nbsp;</td>
    <td class="t ld"><span class="delete">{ldr}</span></td>
    <td class="t ld"><span class="insert">{ldp}</span></td>
    <td class="t">&nbsp;</td>
    <td class="vl">&nbsp;</td>
</tr>
'''.format(
        ldr=get_lex_summ(bookr, my_own_lex),
        ldp=get_lex_summ(bookp, my_own_lex),
    )

def print_table(hp):
    result = '''
<table class="t {}">
'''.format(hp)

    for (i, (
                (bookr, (ch_r_f, vs_r_f), (ch_r_t, vs_r_t)), 
                (bookp, (ch_p_f, vs_p_f), (ch_p_t, vs_p_t)),
    )) in enumerate(corresp_spec):
        result += '''
<tr class="t tb bb">
    <th class="t" colspan="3">{} {}:{}-{}:{}</th>
    <th class="t" colspan="3">{} {}:{}-{}:{}</th>
</tr>
'''.format(bookr, ch_r_f, vs_r_f, ch_r_t, vs_r_t, bookp, ch_p_f, vs_p_f, ch_p_t, vs_p_t)
        result += print_passage(normalized_passages[i], hp)
        result += print_lexeme_summary(bookr, bookp, own_lexemes[i])
    
    result += '''
</table>
'''
    return result

def shin(x): return x.replace(
        '\uFB2A'
        ,'ש'
).replace(
        '\uFB2B',
        'ש'
)

def lines_chapter_mt(ch):
    vn = passage2vnode[('Jesaia', ch, 1)]
    cn = L.u('chapter', vn)
    lines = []
    for v in L.d('verse', cn):
        vl = F.verse.v(v)
        text = ''.join('{}{}'.format(
            F.g_cons_utf8.v(w), ' ' if len(F.trailer_utf8.v(w)) else '') for w in L.d('word', v))
        lines.append('{} {}'.format(vl, shin(text.strip())))
    return lines

def lines_chapter_1q(ch):
    lines = []
    for v in q1[ch]:
        text = ' '.join(q1[ch][v])
        lines.append('{} {}'.format(v, shin(text.strip())))
    return lines

def compare_chapters(c1, c2, lb1, lb2):
    dh = difflib.HtmlDiff(wrapcolumn=60)
    table_html = dh.make_table(
        c1, 
        c2, 
        fromdesc=lb1, 
        todesc=lb2, 
        context=False, 
        numlines=5,
    )
    htext = '''<html>{}<body>{}</body></html>'''.format(diffhead, table_html)
    return htext

def mt1q_chapter_diff(ch):
    lines_mt = lines_chapter_mt(ch)
    lines_1q = lines_chapter_1q(ch)
    return compare_chapters(lines_mt, lines_1q, 'Jesaia {} MT'.format(ch), 'Jesaia {} 1Q'.format(ch))

html_text_h = html_file_tpl.format(
    '2 Kings 19-26 and parallels [Hebrew]',
    css,
    print_table('h'),
)
html_text_p = html_file_tpl.format(
    '2 Kings 19-26 and parallels [phonetic]',
    css,
    print_table('p'),
)
ht = open('kings_parallels_h.html', 'w')
ht.write(html_text_h)
ht.close()
ht = open('kings_parallels_p.html', 'w')
ht.write(html_text_p)
ht.close()

# the Jesaja chapters with 1Q comparison
for ch in range(37,40):
    ht = open('jesaia-mt-1q_{}.html'.format(ch), 'w')
    ht.write(mt1q_chapter_diff(ch))
    ht.close()

HTML(css)

HTML(html_text_h)



