from functools import partial

def power(base,exponent):
    return base**exponent

#Now create partials for sq,cube
sq   = partial(power,exponent=2)
cube = partial(power,exponent=3)

print(sq(10))
print(sq(20))



