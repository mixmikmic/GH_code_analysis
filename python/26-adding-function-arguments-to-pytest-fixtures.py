# Let's create a function that takes every argument
def foo(*args, **kwargs):
    return (args, kwargs)

foo('a', 'b')

foo('a', b=2, c='test')

