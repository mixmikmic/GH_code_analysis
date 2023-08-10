def make_list(n):
    if True:
        return list(range(n))
    else:
        return list(str(i) for i in range(n))

n = int(25e6)
m = int(n*.9)
m = n // 2
a_list = make_list(n)
a_set = set(a_list)

# Finding something that is in a set is fast.
get_ipython().magic('timeit m in a_set')
# Finding something that is _not_ in a set is also fast.
get_ipython().magic('timeit n+1 in a_set')

# Finding something in a list can be slow.
# It starts at the beginning and compares each value.
# The search time depends on where the value is in the list.
get_ipython().magic('timeit m in a_list')
# Finding something that is not is a list is the worst case.
# It has to be compared to all values of the list.
get_ipython().magic('timeit n+1 in a_list')

max_exponent = 6

for n in (10 ** i for i in range(1, max_exponent+1)):
    m = n // 2
    print('%d:' % n)
    a_list = make_list(n)
    a_set = set(a_list)

    get_ipython().magic('timeit m in a_set')

for n in (10 ** i for i in range(1, max_exponent+1)):
    m = n // 2
    print('%d:' % n)
    a_list = make_list(n)
    a_set = set(a_list)

    get_ipython().magic('timeit m in a_list')

def set_foo(n):
    for i in range(n):
        i in a_set

def list_foo(n):
    for i in range(n):
        i in a_list

for n in (10 ** i for i in range(1, max_exponent+1)):
    print('%d:' % n)
    a_list = list(range(n))
    a_set = set(a_list)

    get_ipython().magic('timeit set_foo(n)')

for n in (10 ** i for i in range(1, max_exponent+1)):
    print('%d:' % n)
    a_list = list(range(n))
    a_set = set(a_list)

    get_ipython().magic('timeit list_foo(n)')

