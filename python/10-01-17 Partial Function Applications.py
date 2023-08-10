from functools import partial

def multiply(x, y):
    return x* y

double = partial(multiply, y=2)

double(3
      )

