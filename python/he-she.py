import sys
import collections
from IPython.display import display_pretty, display_html

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

API = fabric.load('etcbc4b', '--', 'he-she', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype
        language lex g_cons g_word_utf8 trailer_utf8
        book chapter verse label number
    ''',''),
    "prepare": prepare,
    "primary": False,
}, verbose='NORMAL')
exec(fabric.localnames.format(var='fabric'))

msg('Collecting all verses with he or she in it')
target_verses = collections.OrderedDict()
mismatch_verses = collections.OrderedDict()
target_lexemes = {'HJ>', 'HW>'}
for w in F.otype.s('word'):
    wl = F.lex.v(w)
    if wl in target_lexemes:
        wg = F.g_cons.v(w)
        lm = wl == 'HW>'
        lf = wl == 'HJ>'
        gm = 'W' in wg
        gf = 'J' in wg
        cls = None
        if lm and not gm or lf and not gf:
            cls = 'x'
        elif lm: cls = 'm'
        elif lf: cls = 'f'
        vs = L.u('verse', w)
        target_verses.setdefault(vs, []).append(wl)
        if cls == 'x':
            mismatch_verses.setdefault(vs, []).append(wl)
        
msg('Done')
mverses = collections.Counter()
tverses = collections.Counter()

totocc = 0
for v in mismatch_verses:
    mverses[len(mismatch_verses[v])] += 1
for (lab, n) in sorted(mverses.items(), key=lambda y: (-y[0])):
    print('Verses with {:>2} mis-occurrences: {:>4}'.format(lab, n))
    totocc += lab * n
print('Verses: {:>5}; mis-occurrences: {:>4}'.format(len(mismatch_verses), totocc))
totocc = 0
for v in target_verses:
    tverses[len(target_verses[v])] += 1
for (lab, n) in sorted(tverses.items(), key=lambda y: (-y[0])):
    print('Verses with {:>2} occurrences: {:>4}'.format(lab, n))
    totocc += lab * n
print('Verses: {:>5}; occurrences: {:>4}'.format(len(target_verses), totocc))

from IPython.display import HTML

css = '''
<style>
td.vl {
    font-family: Verdana, Arial, sans-serif;
    font-size: small;
    text-align: right;
    color: #aaaaaa;
    width: 10%;
}
td.ht {
    font-family: Ezra SIL, SBL Hebrew, Verdana, sans-serif;
    font-size: x-large;
    line-height: 1.7;
    text-align: right;
    direction: rtl;
}
table.ht {
    width: 100%;
    direction: rtl;
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
</style>
'''

def print_verse(v):
    verse_label = '<td class="vl">{} {}:{}</td>'.format(
        F.book.v(L.u('book', v)),
        F.chapter.v(L.u('chapter', v)),
        F.verse.v(v),
    )
    words = []
    for w in L.d('word', v):
        wl = F.lex.v(w)
        wg = F.g_cons.v(w)
        lm = wl == 'HW>'
        lf = wl == 'HJ>'
        gm = 'W' in wg
        gf = 'J' in wg
        cls = None
        if lm and not gm or lf and not gf:
            cls = 'x'
        elif lm: cls = 'm'
        elif lf: cls = 'f'
        wt = F.g_word_utf8.v(w)+F.trailer_utf8.v(w) if cls == None else '<span class="{}">{}</span>{}'.format(
            cls, F.g_word_utf8.v(w), F.trailer_utf8.v(w),
        )
        words.append(wt)
    text = '{}<td class="ht">{}</td>'.format(verse_label, ''.join(words))
    return '<tr class="ht">{}</tr>'.format(text)

def print_verses(vv):
    return '<table class="ht">{}</table'.format(''.join(print_verse(v) for v in vv)) 

HTML(css)

HTML(print_verses(mismatch_verses))



