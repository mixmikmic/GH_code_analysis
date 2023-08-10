squares = []

for x in range(100):
    squares.append(x*x)

squares = [x*x for x in range(100)]

combos = []

for x in [1, 2, 3]:
    for y in [1, 2, 3]:
        if x != y:
            combos.append( (x, y) )
            
combos

combos = [(x, y) for x in [1, 2, 3] for y in [1, 2, 3] if x != y]
combos

combos = [(x, y) for x in [1, 2, 3]
                 for y in [1, 2, 3]
                 if x != y]
combos

words = "The quick brown fox jumped over the lazy dog".split()
words

word_len = {}

for w in words:
    word_len[w] = len(w)
    
word_len

word_len = {w: len(w) for w in words}
word_len

squares = (x*x for x in range(10))
squares

for item in squares:
    print(item)

# map() example
def f(x):
    return x*x

numbers = range(10)

# squares = (f(x) for x in numbers)
squares = map(f, numbers)

squares

list(squares)

# map() over multiple sequences
def add_them(a, b):
    return a + b

seq_a = [2, 4, 6]
seq_b = [1, 2, 3]

results = map(add_them, seq_a, seq_b)

list(results)

# reduce() example
from functools import reduce

seq = [1, 2, 3, 4]

results = reduce(add_them, seq)

results

# filter() example
def is_even(x):
    return x % 2 == 0

seq = range(10)

even_numbers = filter(is_even, seq)

list(even_numbers)

seq = range(10)

even_numbers = (x for x in seq if is_even(x))

list(even_numbers)



