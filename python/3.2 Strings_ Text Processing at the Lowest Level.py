monty = 'Monty Python'

circus = "Monty Python's Flying Circus"
circus

couplet = "Shall I compare thee to a Summer's day?"          "Thou are more lovely and more temperate:"
print(couplet)

couplet = ("Rough winds do shake the darling buds of May,"
           "And Summer's lease hath all too short a date:")
print(couplet)

couplet = """Shall I compare thee to a Summer's day?
Thou are more lovely and more temperate:"""
print(couplet)

couplet = '''Rough winds do shake the darling buds of May,
And Summer's lease hath all too short a date:'''
print(couplet)

'very' + 'very' + 'very'

'very' * 3

a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
for line in b:
    print(line)

print(monty)

grail = 'Holy Grail'
print(monty + grail)

print(monty, grail)

print(monty, "and the", grail)

monty[0]

monty[5]

monty[-1]

## 5 = len(monty) - 7
monty[-7]

sent = 'colorless green ideas sleep furiously'
for char in sent:
    print(char, end=' ')

import nltk
from nltk.corpus import gutenberg
raw = gutenberg.raw('melville-moby_dick.txt')
fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())
fdist.most_common(5)

[char for (char, count) in fdist.most_common()]

fdist.plot()

monty[6:10]

monty[-12:-7]

monty[:5]

monty[6:]

phrase = 'And now for something completely different'
if 'thing' in phrase:
    print('found "thing"')

monty.find('Python')

':'.join(['esto','aquello'])

'dos fuerzas'.replace('dos','una sola').replace('fuerzas','fuerza')

query = 'Who knows?'
beatles = ['John', 'Paul', 'George', 'Ringo']

query[2]

beatles[2]

query[:2]

beatles[:2]

query + " I don't"

beatles + 'Brian'

beatles + ['Brian']

beatles[0] = "John Lennon"
del beatles[-1]
beatles

query[0] = 'F'



