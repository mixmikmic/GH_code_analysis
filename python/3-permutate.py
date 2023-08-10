def permutate(collection):
    '''Yields all the permutations of the collection.
    The collection must be sliceable.'''
    for i, item in enumerate(collection):
        subcollection = collection[:i] + collection[i+1:]
        if subcollection:
            for permutation in permutate(subcollection):
                yield item + permutation
        else:
            yield item

list(permutate('RGB'))

from itertools import permutations

list(permutations('RGB'))

list(permutations([2, 3, 5]))

list(permutations([(2,), (3,), (5,)]))

list(permutations([2]))

def permutate(collection):
    for i, item in enumerate(collection):
        subcollection = collection[:i] + collection[i+1:]
        if subcollection:
            for permutation in permutate(subcollection):
                yield (item,) + permutation
        else:
            yield (item,)

# Ensure that my permutate yields the same output as permutations from itertools.
# My tests are nowhere exhaustive.

test_cases = (
    'RGB',
    [(2,), (3,), (5,)],
    [2, 3, 5],
    [2],
)

def test():
    for collection in test_cases:
        assert list(permutate(collection)) == list(permutations(collection)), (
            collection, list(permutate(collection)), list(permutations(collection)))
    return 'All tests passed'

test()

