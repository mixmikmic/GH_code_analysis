def square_root(x): return math.sqrt(x) # standard function definition

square_root = lambda x: math.sqrt(x) # equivalent definition with lambda

a = [2, 18, 9, 22, 17, 24, 8, 12, 27]

print filter(lambda x: x % 3 == 0, a)

print map(lambda x: x * 2 + 10, a)

print reduce(lambda x, y: x + y, a)

