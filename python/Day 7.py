import numpy as np

np.sin(30)

calculated = []
for deg in range(0, 91):
    calculated.append(np.sin(deg))
calculated

calculated2 = [np.sin(deg) for deg in range(0, 91)]
calculated2

np.sin(range(0, 91))

np.sin([deg for deg in range(0, 91)])

get_ipython().magic('timeit [np.sin(deg) for deg in range(0, 91)]')

get_ipython().magic('timeit np.sin([deg for deg in range(0, 91)])')



paragraph = """
The commencement of the move for a scientific standard of length in
France which resulted in the mètre was in 1790, when the revolutionary
government proposed to England the formation of a commission of equal
numbers from the English Royal Society and the French Academy, for the
purpose of fixing the length of the seconds pendulum at latitude 45° as
the basis of a new system of measures. This proposal was not favorably
received, and the Academy, at the request of government, appointed as
a commission Borda, Lagrange, Laplace, Monge, and Condorcet, to decide
whether the seconds pendulum, the quarter of the equator, or the quarter
of a meridian, should be used as the natural standard for the new system
of measures. They settled on the last as best for the purpose, and
resolved that the ten millionth of the meridian quadrant, or distance
from equator to pole, measured at sea level, be taken for basis of the
new system, and be called a mètre.
"""

cleaned_text = paragraph.lower().replace(',', '').    replace('.', '').replace('\n', ' ').strip()

cleaned_text

set(cleaned_text)

cleaned_text.split()

set(cleaned_text.split())

dir([])

help([].count)

word_list = cleaned_text.split()

word_set = set(word_list)

{word: word_list.count(word) for word in word_set}

type([])

type({})

type(set())

dir('')

'45°'.isdigit()

help(''.isdigit)

help(''.isalnum)

help(''.isnumeric)

help(''.isdecimal)

help(''.isalpha)

{word for word in word_list if word.isalpha()}

{word if word.isalpha() else '' for word in word_list}

dir(set())

{5, 6, 7, 8, 9, 9}

abc = {4, 5, 6, 6, 7, 7, 8, 9, 9}

abc

abc.add(4)

abc

pqr = {7, 8, 9}

pqr

abc.update(pqr)

abc.update({9, 1, 3})

abc

abc.issubset(pqr)

pqr.issubset(abc)

abc

abc.pop()

abc

abc.remove(3)

abc

help(abc.pop)

lst = [7, 8, 9]

lst.append(6)

lst

dir(lst)

help(lst.insert)

lst

lst.insert(1, 3)

lst

lst.pop()

lst

lst.pop()

lst.pop(0)

help(lst.pop)



dct = {'a': 1, 'b': 3}

dct

dct['a']

dct['c'] = 5

dct

dir(dct)

dct

dct.update({'c': 7, 'd': 6})

dct

dct.values()

list(dct.values())

[x * 2 for x in dct.values()]



dct.keys()



dct.items()

for item in dct:
    print("Key {} Value {}".format(item, dct[item]))

for key, value in dct.items():
    print("Key {} Value {}".format(key, value))

for item in dct.items():
    print(item)

name = 'Harry'
age = 16

name

age

name, age = 'Ron', 16

name

age

name, age = ('Gadalf', 2500)

name

age

(name, age) = ('Gimli', 230)

name

a, b = 0, 1, 2

a, b, c = 0, 1

for item in dct.items():
    print(item)

for item in dct.items():
    key, value = item
    print(key, '---', value)

for key, value in dct.items():
    print(key, '---', value)



