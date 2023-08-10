def make_list(n):
    if True:
        return list(range(n))
    else:
        return list(str(i) for i in range(n))

n = int(25e6)
# n = 5
m = (0, n // 2, n-1, n)
a_list = make_list(n)
a_set = set(a_list)

n, m

# Finding something that is in a set is fast.
# The key one is looking for has little effect on the speed.
beginning = 0
middle = n//2
end = n-1
get_ipython().magic('timeit beginning in a_set')
get_ipython().magic('timeit middle in a_set')
get_ipython().magic('timeit end in a_set')
# Finding something that is _not_ in a set is also fast.
get_ipython().magic('timeit n in a_set')

# Searching for something in a list
# starts at the beginning and compares each value.
# The search time depends on where the value is in the list.
# That can be slow.
beginning = 0
middle = n//2
end = n-1
get_ipython().magic('timeit beginning in a_list')
get_ipython().magic('timeit middle in a_list')
get_ipython().magic('timeit end in a_list')
# Finding something that is not is a list is the worst case.
# It has to be compared to all values of the list.
get_ipython().magic('timeit n in a_list')

max_exponent = 6

for n in (10 ** i for i in range(1, max_exponent+1)):
    a_list = make_list(n)
    a_set = set(a_list)

    m = (0, n // 2, n-1, n)
    for j in m:
        print('length is %s, looking for %s' % (n, j))
        get_ipython().magic('timeit j in a_set')

for n in (10 ** i for i in range(1, max_exponent+1)):
    a_list = make_list(n)
    a_set = set(a_list)

    m = (0, n // 2, n-1, n)
    for j in m:
        print('length is %s, looking for %s' % (n, j))
        get_ipython().magic('timeit j in a_list')

