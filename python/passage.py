import sys,os,re
import collections
from IPython.display import HTML, display_pretty, display_html

import laf
from laf.fabric import LafFabric
from etcbc.lib import Transcription
from etcbc.preprocess import prep

fabric = LafFabric()

source = 'etcbc'
versions = ('4b', '4c')

FF = {}
MSG = {}
LL = {}
for version in ('4b', '4c'):
    API = fabric.load(source+version, '--', 'passage', {
        "xmlids": {"node": False, "edge": False},
        "features": ('''
            otype
            g_cons g_word g_cons_utf8 g_word_utf8 g_word trailer_utf8
            book chapter verse label
        ''',''),
        "prepare": prep(select={'L'}),
        "primary": False,
    }, verbose='NORMAL')
    FF[version] = API['F']
    MSG[version] = API['msg']
    LL[version] = API['L']

verses = {}
for version in versions:
    msg = MSG[version]
    F = FF[version]
    msg("{}: Making a mapping between a passage specification and a verse node".format(version))
    versesv = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))
    for vn in F.otype.s('verse'):
        bk = F.book.v(vn)
        ch = int(F.chapter.v(vn))
        vs = int(F.verse.v(vn))
        versesv[bk][ch][vs] = vn
    verses[version] = versesv
    msg('Done')

HTML('''
<style type="text/css">
td.ht {
    font-family: Ezra SIL, SBL Hebrew, Verdana, sans-serif;
    font-size: x-large;
    line-height: 1.7;
    text-align: right;
    direction: rtl;
}
td.et {
    font-family: Verdana, sans-serif;
    font-size: medium;
    line-height: 1.2;
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
}
</style>
''')

accent_pat = re.compile('[*0-9]')
tr = Transcription()

def print_verse(bk, ch, vs, vowels=True, accents=True):
    rows = {}
    for version in versions:
        F = FF[version]
        L = LL[version]
        label = '{} {}:{}'.format(bk, ch, vs)
        vn = verses[version][bk][ch][vs]
        treps = []
        trepes = []
        for w in L.d('word', vn):
            if not vowels:
                trep = '{}{}'.format(F.g_cons_utf8.v(w), F.trailer_utf8.v(w))
                trepe = F.g_cons.v(w)
            else:
                trep = '{}{}'.format(F.g_word_utf8.v(w), F.trailer_utf8.v(w))
                trepe = F.g_word.v(w)
                if not accents:
                    trep = Transcription.to_hebrew(accent_pat.sub('', tr.from_hebrew(trep)))
            treps.append(trep)
            trepes.append(trepe)
        text = ''.join(treps)
        texte = ' '.join(trepes)
        rows[version] = '''
    <tr><td class="vl">{}</td><td class="ht">{}</td></tr>
    <tr><td class="vl">{}</td><td class="et">{}</td></tr>
        '''.format(version, text, label, texte)
    return '''
<table>
    {}
</table>'''.format('\n'.join(rows[version] for version in versions))

pc = lambda bk, ch, vs: print_verse(bk, ch, vs, vowels=False, accents=False)    # no vowels, no accents
pv = lambda bk, ch, vs: print_verse(bk, ch, vs, vowels=True,  accents=False)    # vowels, no accents
pa = lambda bk, ch, vs: print_verse(bk, ch, vs, vowels=True,  accents=True)     # vowels and accents

HTML(pc('Esther', 3, 4))

HTML(pv('Esther', 3, 4))

HTML(pa('Esther', 3, 4))



