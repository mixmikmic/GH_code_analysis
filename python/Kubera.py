import operator

def Range(start, stop=None, step=1):
    """
    Get range generator.
    """
    # the start value is actually stop - swap them
    if stop is None:
        start, stop = 0, int(start)
    # ensure all variables are integers
    start, stop, step = map(int, (start, stop, step))
    
    if step < 0:
        cmp = operator.gt
    elif step > 0:
        cmp = operator.lt
    else:
        raise ValueError("Third argument must NOT be zero")

    i = start
    while cmp(i, stop):
        yield i
        i += step

Range(5)

list(Range(0, 5)), list(Range(-5, 0))

list(Range(0, -5, -1))

list(Range(0, 10, 2)), list(Range(0, -10, -2))

import math

def sieve(n):
    # create list of n booleans indicating whether index
    # is prime. 0 and 1 are automatically not prime. The
    # rest are true
    res = [False, False] + [True] * (n - 2)
    
    # function returning iterator (filter) of all indexes
    # which are True
    get_true_items = lambda *args: filter(res.__getitem__, Range(*args))

    # Only have to loop through first √n items
    sqrt = int(math.sqrt(n))
    
    # Set all composite indices to False
    for i in get_true_items(sqrt + 1):
        yield i # the next 'true' item is prime
        for j in Range((n + 1) // i - i):
            res[i * (i + j)] = False

    # yield the rest of the true items
    yield from get_true_items(sqrt + 1, n)

list(sieve(20))

list(sieve(200)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

import itertools
list(itertools.permutations("abc"))

def simple_permute(string):
    for permutation in itertools.permutations(string):
        yield ''.join(permutation)

list(simple_permute('abc'))

def permute_string(string):
    if len(string) == 0:
        return
    elif len(string) == 1:
        yield string
    else:
        a = string[0]
        for p in permute_string(string[1:]):
            for i in range(len(p) + 1):
                yield "{}{}{}".format(p[:i], a, p[i:])

list(permute_string('a'))

list(permute_string('ab'))

list(permute_string('abc'))

len(list(permute_string('abcd'))) == math.factorial(4)

list(permute_string('aaa'))


def p_triples(start=3, stop=100):
    for a in range(start, stop):
        for b in range(a + 1, stop):
            s = math.sqrt(a ** 2 + b ** 2)
            c = int(s)
            if s == c:
                # check if this number is factorable (ignore)
                x = math.gcd(a, b)
                if x != 1 and c / x == c // x:
                    continue

                yield (a, b, c)

list(filter(lambda x: x[2] < 100, p_triples(3, 250)))

def euclid_triples(stop=600):
    for n in range(1, stop):
        for m in range(n + 1, stop):
            a = m ** 2 - n ** 2
            b = 2 * m * n
            c =  m ** 2 + n ** 2
            x = math.gcd(a, b)
            if x != 1 and c / x == c // x:
                continue
            assert c == math.sqrt(a ** 2 + b ** 2)
            yield (a, b, c)

sorted(euclid_triples(10), key=lambda x: x[2])



















