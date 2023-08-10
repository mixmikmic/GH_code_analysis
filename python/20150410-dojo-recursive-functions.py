def triangle(n):
    total = n
    if n > 0:
        total += triangle(n - 1)
    return total


triangle(4)

def triangle(n):
    if n <= 0:
        return 0
    return n + triangle(n - 1)

triangle(4)

def triangle(n):
    if n > 0:
        return n + triangle(n - 1)
    return 0

triangle(4)

def triangle(n):
    return n + (triangle(n-1) if n>0 else 0)

triangle(4)

import sys

sys.getrecursionlimit()

sys.setrecursionlimit(100000)

sys.getrecursionlimit()

