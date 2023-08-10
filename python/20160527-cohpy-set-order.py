{'hello': 324, 'world': 98743, 'city': None}

b = {'hello', 'world', 'city', 'zebra', 'aardvark', 'Aaron', 'aaa', 'foo', 'xray', 'alex', 'jim', 'chris'}

words = set(open('/usr/share/dict/american-english').read().split())
type(words), len(words)

from itertools import islice

b = set()
for word in islice(words, 999):
    b |= set([word])
    # print(word)
    
b

str(b)

repr(b)

print(b)

print(str(b))

print(repr(b))

for word in b:
    print(repr(word))

b = set()
for word in islice(words, 1000):
    b |= set([word])
    # print(word)
    
b

