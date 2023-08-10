colors = ['branco', 'amarelo', 'azul', 'branco']

set(colors)

a = 'maça'
print len(a)

b = u'maça'
print len(b)

a == b

# Your problem may be solved adding the directive:
from __future__ import unicode_literals

a = 'maça'
b = u'maça'
a == b

a = u'maça'.encode('latin1')

print a

unicode(a)

# Even simple concatenation raises a decode problem
a + 'outra palavra'

# some libraries can help, but remember: find the correct encoding is a heuristic science!
import chardet
chardet.detect(a)

print a.decode(chardet.detect(a)['encoding'])

wordlist = ['abacate', 'kiwi', 'abacaxi', 'melancia']
[word for word in wordlist if word.startswith('a')]

wordlist = ['abacate', 'kiwi', 'abacaxi', 'melancia', 'banana', 'abacate', 'abacaxi', 'abacate']

# join, sorted, set
print 'Frutas: ' + ', '.join( sorted(set(wordlist)))

# use always operator 'in', instead of a loop with ==
'abacaxi' in wordlist

# Frequency lists
freqlist = dict()
for word in wordlist:
    freqlist[word] = freqlist.get(word, 0) + 1
print freqlist

# sorted by frequency
from operator import itemgetter
', '.join( [word for word,freq in sorted(freqlist.items(), key=itemgetter(1), reverse=True)] )

# use and abuse of slicing
text = 'E ele disse: "texto em quotes" e continou...'
text[text.find('"')+1:text.rfind('"')]

# Counter
from collections import Counter
Counter(wordlist)

# blist - The blist is a drop-in replacement for the Python list that provides 
# better performance when modifying large lists. 
from blist import blist
blist(wordlist)

# String data in a MARISA-trie may take up to 50x-100x less memory than 
# in a standard Python dict; the raw lookup speed is comparable; 
# trie also provides fast advanced methods like prefix search.
import marisa_trie
trie = marisa_trie.Trie(wordlist)
trie.items()

'abacate' in trie

trie.keys('aba')

