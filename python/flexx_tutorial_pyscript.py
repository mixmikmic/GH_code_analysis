from flexx.pyscript import py2js, evalpy

js = py2js('for i in range(10): print(i)')
print(js)

def foo(x):
    res = []
    for i in range(x):
        res.append(i**2)
    return res
js = py2js(foo)
print(js)

def foo(x):
    return [i**2 for i in range(x)]
js = py2js(foo)
print(js)

class Bar:
    def spam(self):
        return 3 + 4
#js = py2js(Bar)
# This only works if Bar is defined in an actual module.

evalpy('print(3 + 4)')

evalpy('print(None)')



