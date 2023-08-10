def gen_even_fibonacci(last):
    a = 0
    b = 1
    while True:
        c = a + b
        a = b
        b = c
        if b > last:
            break
        if b % 2 == 0:
            yield b

[i for i in gen_even_fibonacci(1000)]

def gen_fibonacci():
    a, b = 0, 1
    while True:
        a, b = b, a + b
        yield b

def gen_even(gen):
    for i in gen:
        if i % 2 == 0:
            yield i
                

def gen_lte(gen, n):
    for i in gen:
        if i > n:
            break
        yield i

[i for i in gen_lte(gen_even(gen_fibonacci()), 1000)]

[i for i in gen_lte(gen_fibonacci(), 1000)]

evens = (i for i in gen_fibonacci() if i%2 == 0)
[i for i in gen_lte(evens, 1000)]

evens = gen_even(gen_fibonacci())
[i for i in gen_lte(evens, 1000)]

[i for i in gen_lte(gen_even(gen_fibonacci()), 1000)]

def gen_n(gen, n):
    for i in gen:
        if n <= 0:
            break
        yield i
        n -= 1

[i for i in gen_n(gen_fibonacci(), 10)]

[i for i in gen_n(gen_even(gen_fibonacci()), 10)]

fibs = gen_fibonacci()
evens = gen_even(fibs)
lte = gen_lte(evens, 100)
[i for i in lte]

