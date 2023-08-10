from itertools import islice, accumulate, count, repeat

n = 12

list(islice(repeat(1), n))

list(islice(accumulate(repeat(1)), n))

list(range(1, n+1))

list(islice(count(1), n))

def range1(n):
    return range(1, n+1)

def count1():
    return count(1)

list(accumulate(range1(n)))

list(islice(accumulate(count1()), n))

list(accumulate(accumulate(range1(n))))

list(islice(accumulate(accumulate(count1())), n))

list(accumulate(accumulate(accumulate(range1(n)))))

list(islice(accumulate(accumulate(accumulate(count1()))), n))

