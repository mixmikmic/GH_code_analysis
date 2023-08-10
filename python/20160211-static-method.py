from __future__ import print_function

class C(object):
    @staticmethod
    def f(*args):
        print('f', repr(args))

    def g(*args):
        print('g', repr(args))

C.f('hello', 'world')
C.g('whirled', 'peas')

c = C()
c.f('Halo', 'mundo')
c.g('girando', 'pisum')

