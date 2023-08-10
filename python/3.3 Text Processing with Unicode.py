import nltk
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line)

f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

ord('ń')

hex(324)

nacute = '\u0144'
nacute

nacute.encode('utf8')

import unicodedata
lines = open(path, encoding='latin2').readlines()
line = lines[2]
print(line.encode('unicode_escape'))

for c in line:
    if ord(c) > 127:
        print('{} U+{:04x} {}'.format(c.encode('utf8'), ord(c), unicodedata.name(c)))

for c in line:
    if ord(c) > 127:
        print('{} U+{:04x} {}'.format(c, ord(c), unicodedata.name(c)))

line.find('zosta\u0142y')

line = line.lower()
line

line.encode('unicode_escape')

import re
m = re.search('\u015b\w*', line)
m.group()

from nltk import word_tokenize
word_tokenize(line)



