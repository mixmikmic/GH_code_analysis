from __future__ import unicode_literals
import codecs

with codecs.open('LIWC2007_Portugues_win.dic', encoding='latin1') as fp:
        liwc = fp.readlines()

pos = set()
neg = set()
for line in liwc:
    line = line.strip()
    if line:
        line = line.split()
        word = line[0]
        cat = line[1:]
        if '126' in cat:
            pos.add(word)
        elif '127' in cat:
            neg.add(word)      
        

len(neg)

len(pos)

text = 'não gostei muito do meu iphone'

text.split()

def check_pol(text):
    pol = 0
    text = text.split()
    negation = False
    if u'não' in text:
        negation = True
    for token in text:
        if token in pos:
            pol += 1
            print 'pos: ' + token
        if token in neg:
            print 'neg: ' + token
            pol += -1
    if negation:
        pol = -1 * pol
    return pol

text = 'Eu gostei muito deste lindo aparelho'
check_pol(text)



