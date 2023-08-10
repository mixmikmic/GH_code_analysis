class a:
    def __init__(self, func):
        self.func = func
        print('a init')

    def __call__(self, *args, **kwargs):
        print('a before calling {0}'.format(self.func))
        result = self.func(*args, **kwargs)
        print('a after calling {0}'.format(self.func))
        return result

class b:
    def __init__(self, func):
        self.func = func
        print('b init')

    def __call__(self, *args, **kwargs):
        print('b before calling  {0}'.format(self.func))
        result = self.func(*args, **kwargs)
        print('b after calling {0}'.format(self.func))
        return result

@a
@b
def f(n):
    return n*n

f(4)

def g(n):
    return n*n

a(b(g))(4)

class a:
    def __init__(self, *args):
        self.decargs = args
        print('a init')

    def __call__(self, func):
        print('a __call__ invoked to return wrapped function')
        def wrapped(*args, **kwargs):
            print('a before calling {0}'.format(func))
            result = func(*args, **kwargs)
            print('a after calling {0}'.format(func))
            return result
        return wrapped

class b:
    def __init__(self, *args):
        self.decargs = args
        print('b init')

    def __call__(self, func):
        print('b __call__ invoked to return wrapped function')
        def wrapped(*args, **kwargs):
            print('b before calling  {0}'.format(func))
            result = func(*args, **kwargs)
            print('b after calling {0}'.format(func))
            return result

        return wrapped

@a(1,2)
@b(3,4)
def f(n):
    return n*n

f(4)

def g(n):
    return n*n

a(1,2)(b(3,4)(g))(4)

f(4)

