def fib(n):
    n1 = 1
    n2 = 2
    tot = 2
    while n2 < n:
        temp = n2
        n2 = n1 + n2
        n1 = temp
        if n2 % 2 == 0:
            tot += n2
    return tot

fib(4_000_000)

