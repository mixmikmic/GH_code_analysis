foo = 'Monty'
bar = foo
foo = 'Python'
bar

foo = ['Monty', 'Python']
bar = foo
foo[1] = 'Bodkin'
bar

empty = []
nested = [empty, empty, empty]
nested

empty = []
nested = [empty, empty, empty]
nested[1].append('Python')
nested[2].append('Ou')
nested

nested = [['a']] * 3
nested

print( id(nested[0]), id(nested[1]), id(nested[2]) )

nested = [['a']] * 3
nested[1].append('Python')
nested[1] = ['Monty']
nested

id(nested)

n2 = nested[:]
id(n2)

print( id(nested[0]), id(n2[2]) )

n2[2].append('lila')

nested[2]

import copy
n3 = copy.deepcopy(nested)

print( id(n3[0]) )

n3[0]

size = 5
python = ['Python']
snake_nest = [python] * size
snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]

id(snake_nest[0])

id(snake_nest[1])

snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]

import random
position = random.choice(range(size))
snake_nest[position] = ['Python']
snake_nest

snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]

snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]

position

id(snake_nest[0])

print( id(snake_nest[1]), id(snake_nest[3]) )

[id(snake) for snake in snake_nest]

mixed = ['cat', '', ['dog'], []]
for element in mixed:
    if element:
        print(element)

sent = ['No', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']
all(len(w) > 4 for w in sent)

any(len(w) > 4 for w in sent)



