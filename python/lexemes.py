import sys, collections, re

from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
fabric.load('etcbc{}'.format(version), 'lexicon', 'lexemes', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype
        lex lex_utf8 g_entry_heb
        sp gloss
        book chapter verse
    ''',''),
    "prepare": prepare,
    "primary": False,
})
exec(fabric.localnames.format(var='fabric'))

csvdir = my_file('csv')
passagedir = my_file('passage')
get_ipython().run_line_magic('mkdir', '-p {csvdir}')
get_ipython().run_line_magic('mkdir', '-p {passagedir}')

passage_pat = re.compile('^\s*([A-Za-z0-9_]+)\s*([0-9,-]*)\s*:?\s*([0-9,-]*)\s*$')

lex_info = {}
lex_section = {}
lex_count = collections.Counter()
for v in F.otype.s('verse'):
    bk = F.book.v(L.u('book', v))
    ch = F.chapter.v(L.u('chapter', v))
    vs = F.verse.v(v)
    for w in L.d('word', v):
        lex = F.lex.v(w)
        if lex not in lex_info:
            lex_info[lex] = (F.lex_utf8.v(w), F.g_entry_heb.v(w), F.sp.v(w), F.gloss.v(w))
        lex_section.setdefault(bk, {}).setdefault(ch, {}).setdefault(vs, collections.Counter())[lex] += 1
        lex_count[lex] += 1

def verse_index():
    result = {}
    for v in F.verse.s():
        bk = F.book.v(L.u('book', v))
        ch = F.chapter.v(L.u('chapter', v))
        vs = F.verse.v(v)
        result.setdefault(bk, {}).setdefault(ch, {})[vs] = v
    return result

vindex = verse_index()

def parse_passages(passages):
    lexemes = set()
    for p in passages.strip().split('|'):
        lexemes |= parse_passage(p.strip())
    return lexemes

def parse_ranges(rangespec, kind, passage, source, subsources=None):
    numbers = set()
    if rangespec == '':
        if subsources == None:
            return set(source.keys())
        else:
            for subsource in subsources:
                if subsource in source:
                    numbers |= set(source[subsource].keys())
            return numbers
    ranges = rangespec.split(',')
    good = True
    for r in ranges:
        comps = r.split('-', 1)
        if len(comps) == 1:
            b = comps[0]
            e = comps[0]
        else:
            (b,e) = comps
        if not (b.isdigit() and e.isdigit()):
            print('Error: Not a valid {} range: [{}] in [{}]'.format(kind, r, passage))
            good = False
        else:
            b = int(b)
            e = int(e)
            for c in range(b, e+1):
                crep = str(c)
                if subsources == None:
                    if crep not in source:
                        print('Warning: No such {}: {} ([{}] in [{}])'.format(kind, crep, r, passage))
                    numbers.add(crep)
                else:
                    for subsource in subsources:
                        if subsource not in source or crep not in source[subsource]:
                            print('Warning: No such {}: {}:{} ([{}] in [{}])'.format(kind, subsource, crep, r, passage))
                    numbers.add(crep)
    return numbers
    
def parse_passage(passage):
    lexemes = set()
    result = passage_pat.match(passage)
    if result == None:
        print('Error: Not a valid passage: {}'.format(passage))
        return lexemes
    (book, chapterspec, versespec) = result.group(1,2,3)
    if book not in vindex:
        print('Error: Not a valid book: {} in {}'.format(book, passage))
        return lexemes
    chapters = parse_ranges(chapterspec, 'chapter', passage, vindex[book])
    verses = parse_ranges(versespec, 'verse', passage, vindex[book], chapters)

    vnodes = set()
    for ch in vindex[book]:
        if ch not in chapters: continue
        for vs in vindex[book][ch]:
            if vs not in verses: continue
            vnodes.add(vindex[book][ch][vs])
    lexemes = set()
    for v in vnodes:
        for w in L.d('word', v):
            lexemes.add(F.lex.v(w))
    return lexemes
        
def lexbase(passages, excluded=None):
    lexemes = parse_passages(passages)
    outlexemes = set() if excluded == None else parse_passages(excluded)
    lexemes -= outlexemes
    fileid = '{}{}'.format(
        passages, 
        '' if excluded == None else ' minus {}'.format(excluded)
    )
    filename = 'passage/{}.csv'.format(fileid.replace(':','_'))
    of = outfile(filename)
    i = 0
    limit = 20
    nlex = len(lexemes)
    shown = min((nlex, limit))
    print('==== {} ==== showing {} of {} lexemes here ===='.format(fileid, shown, nlex))
    for lx in sorted(lexemes, key=lambda x: (-lex_count[x], x)):
        (l_utf8, l_vc, l_sp, l_gl) = lex_info[lx]
        line = '"{}",{},{}","{}","{}","{}"\n'.format(lx, lex_count[lx], l_utf8, l_vc, l_sp, l_gl)
        of.write(line)
        if i < limit: sys.stdout.write(line)
        i += 1
    of.close()
    print('See {}\n'.format(my_file(filename)))

lexbase('Genesis 2', excluded=None)
lexbase('Genesis 2', excluded='Genesis 1')
lexbase('Genesis 3-4,10', excluded='Genesis 1-2')
lexbase('Exodus', excluded='Genesis')
lexbase('Numeri 1-3:10-15|Judices 5:1,3,5,7,9|Ruth 4', excluded='Chronica_I|Chronica_II')

outf = outfile("csv/all_lexemes.csv")
for (l, f) in sorted(lex_count.items(), key=lambda x: -x[1]):
    (l_utf8, l_vc, l_sp, l_gl) = lex_info[l]
    outf.write('"{}",{},"{}","{}","{}","{}"\n'.format(
        l, f, l_utf8, l_vc, l_sp, l_gl,
    ))
outf.close()

for bk in sorted(lex_section):
    outfb = outfile("csv/{}.csv".format(bk))
    outfc = outfile("csv/{}_per_ch.csv".format(bk))
    outfci = outfile("csv/{}_per_ch_inc.csv".format(bk))
    outfv = outfile("csv/{}_per_vs.csv".format(bk))
    outfvi = outfile("csv/{}_per_vs_inc.csv".format(bk))
    bk_lex = set()
    for ch in sorted(lex_section[bk], key=lambda x: int(x)):
        ch_lex = set()
        for vs in sorted(lex_section[bk][ch], key=lambda x: int(x)):
            for l in sorted(lex_section[bk][ch][vs]):
                (l_utf8, l_vc, l_sp, l_gl) = lex_info[l]
                f = lex_count[l]
                line = '"{}",{},{},"{}",{},"{}","{}","{}","{}"\n'.format(
                    bk, ch, vs, l, f, l_utf8, l_vc, l_sp, l_gl,
                )
                outfv.write(line)
                if l not in ch_lex:
                    ch_lex.add(l)
                    outfvi.write(line)
                if l not in bk_lex:
                    bk_lex.add(l)
        for l in sorted(ch_lex):
            (l_utf8, l_vc, l_sp, l_gl) = lex_info[l]
            f = lex_count[l]
            line = '"{}",{},"{}",{},"{}","{}","{}","{}"\n'.format(
                bk, ch, l, f, l_utf8, l_vc, l_sp, l_gl,
            )
            outfc.write(line)
            if l not in bk_lex:
                bk_lex.add(l)
                outfci.write(line)
    for l in sorted(bk_lex):
        (l_utf8, l_vc, l_sp, l_gl) = lex_info[l]
        f = lex_count[l]
        line = '"{}","{}",{},"{}","{}","{}","{}"\n'.format(
            bk, l, f, l_utf8, l_vc, l_sp, l_gl,
        )
        outfb.write(line)
    outfb.close()
    outfc.close()
    outfci.close()                    
    outfv.close()
    outfvi.close()



