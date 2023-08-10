import collections

a= {'a':'A','c':'C'}
b= {'b':'B','c':'D'}

m=collections.ChainMap(a,b)

print('Indivisual Values')
print('a={}'.format(m['a']))
print('b={}'.format(m['b']))
print('c={}'.format(m['c']))
print()

print('Keys={}'.format(list(m.keys())))
print('Value={}'.format(list(m.values())))

print()

print("Print Items")

for k,v in m.items():
    print('{}={}'.format(k,v))
    
print('"d" in m:{}'.format(('d' in m)))

import collections

a = {'a': 'A', 'c': 'C'}
b = {'b': 'B', 'c': 'D'}

m = collections.ChainMap(a, b)

print(m.maps)
print('c = {}\n'.format(m['c']))

# reverse the list
m.maps = list(reversed(m.maps))

print(m.maps)
print('c = {}'.format(m['c']))

import collections

a = {'a': 'A', 'c': 'C'}
b = {'b': 'B', 'c': 'D'}

m = collections.ChainMap(a, b)
print('Before: {}'.format(m['c']))
a['c'] = 'E'
print('After : {}'.format(m['c']))

import collections

a = {'a': 'A', 'c': 'C'}
b = {'b': 'B', 'c': 'D'}

m1 = collections.ChainMap(a, b)
m2 = m1.new_child()

print('m1 before:', m1)
print('m2 before:', m2)

m2['c'] = 'E'

print('m1 after:', m1)
print('m2 after:', m2)

import collections

a = {'a': 'A', 'c': 'C'}
b = {'b': 'B', 'c': 'D'}
c = {'c': 'E'}

m1 = collections.ChainMap(a, b)
m2 = m1.new_child(c)

print('m1["c"] = {}'.format(m1['c']))
print('m2["c"] = {}'.format(m2['c']))

import collections

a = {'a': 'A', 'c': 'C'}
b = {'b': 'B', 'c': 'D'}
c = {'e': 'E', 'f': 'F'}


m1 = collections.ChainMap(a, b,c)
m1.parents



