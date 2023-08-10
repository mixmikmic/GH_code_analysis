import itertools


def smile(l):
    
    """Takes a list of integers. For each integer (i), create
    a list of smileys of length i. Then flatten this list and
    return the result."""

    # This is very functional!
    return list(itertools.chain(*[['☺']*i for i in l]))

# [1,2] → [ ['☺'], ['☺', '☺'] ] → ['☺', '☺', '☺']
print(smile([1,2]))

print('Starting test')
assert(smile([]) == [])
assert(smile([1]) == ['☺'])
assert(smile([0]) == [])
assert(smile([1,0,2]) == ['☺', '☺', '☺'])
print('Done')

def smile(l):
    
    return ['☺'] * sum(l)

print(smile([1,2]))

