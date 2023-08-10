def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

fib(7)

mem = {0:0, 1:1}

def fib_mem(n):
    if n not in mem:
        mem[n] = fib(n - 1) + fib(n - 2)
    return mem[n]

fib_mem(7)

get_ipython().run_line_magic('timeit', 'fib(35)')
# We get 5.54 seconds to run with n=35

get_ipython().run_line_magic('timeit', 'fib_mem(35)')
# We get 412 ns to run with n=35

