def f(i):
    yield i
    while i > 1:
        if i % 2 == 0:
            i //= 2
        else:
            i = 3 * i + 1
        yield i

list(f(22))

def get_stopping_time(i):
    return len(list(f(i)))

assert get_stopping_time(22) == 16

def get_stopping_time(i):
    '''Directly calculates "stopping time",
    without saving the hailstone numbers.'''
    n = 1
    while i > 1:
        if i % 2 == 0:
            i //= 2
        else:
            i = 3 * i + 1
        n += 1
    return n

assert get_stopping_time(22) == 16

def foo(i, j):
    return max(get_stopping_time(k) for k in range(i, j+1))

# From bottom of https://uva.onlinejudge.org/external/1/100.html

sample_input_output = '''1 10 20
100 200 125
201 210 89
900 1000 174'''

for line in sample_input_output.split('\n'):
    i, j, expected_output = [int(s) for s in line.split()]
    actual_output = foo(i, j)
    assert actual_output == expected_output
    print(i, j, actual_output)

