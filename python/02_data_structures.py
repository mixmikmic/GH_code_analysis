groceries = ['eggs', 'milk', 'bacon'] # because, who doesn't like bacon?

# What if we forgot something?
groceries.append('coffee')

# if we don't want something
groceries.remove('eggs')

# and so much more!

L = []          # declare empty list
L.append(1.2)   # add a number 1.2
L.append('a')   # add a text element
L[0] = 1.3      # change an item
del L[1]        # delete an item
len(L)          # length of list
L.count(x)      # count the number of times x occurs
L.index(x)      # return the index of the first occurrence of x
L.remove(x)     # delete the first occurrence of x
L.reverse()     # reverse the order of elements in the list
L = ['a'] + ['b'] # can combine two lists together

print(groceries)
print(groceries[0])
print(groceries[-1])
print(groceries[2])
print(groceries[:2])
print(groceries[-3:])
print(groceries[10])
print(groceries[::2])

d = {'key1':'value', 'key2':20, 'key3':['a', 'list'], 'key5':{'a':'nested','dictionary':'!'}}
d = dict(key1=3, key2='value', key3=True)
print(d['key3'])
print(d.keys())
print(d.items())

