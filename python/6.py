def sum_square(n):
    return sum([x**2 for x in range(1, n+1)])

def square_sum(n):
    return sum([x for x in range(1, n+1)])**2

def diff(n):
    return square_sum(n) - sum_square(n)

get_ipython().magic('timeit diff(100)')



