a = [2, 3, 5, 7]

# Length of a list
len(a)

# Append a value to the end
a.append(11)
a

# Addition concatenates lists
a + [13, 17, 19]

# sort() method sorts in-place
a = [2, 5, 1, 6, 3, 4]
a.sort()
a

a = [1, 'two', 3.14, [0, 3, 5]]
a

a = [2, 3, 5, 7, 11]

a[0]

a[1]

a[-1]

a[-2]

a[0:3]

a[:3]

a[-3:]

a[::2]  # equivalent to a[0:len(a):2]

a[::-1]

a[0] = 100
print(a)

a[1:3] = [55, 56]
print(a)

t = (1, 2, 3)

t = 1, 2, 3
print(t)

len(t)

t[0]

numbers = {'one':1, 'two':2, 'three':3}
# or
numbers = dict(one=1, two=2, three=2)

# Access a value via the key
numbers['two']

# Set a new key:value pair
numbers['ninety'] = 90
print(numbers)

primes = {2, 3, 5, 7}
odds = {1, 3, 5, 7, 9}

a = {1, 1, 2}

a

# union: items appearing in either
primes | odds      # with an operator
primes.union(odds) # equivalently with a method

# intersection: items appearing in both
primes & odds             # with an operator
primes.intersection(odds) # equivalently with a method

# difference: items in primes but not in odds
primes - odds           # with an operator
primes.difference(odds) # equivalently with a method

# symmetric difference: items appearing in only one set
primes ^ odds                     # with an operator
primes.symmetric_difference(odds) # equivalently with a method

