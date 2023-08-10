class Foo(object):
    @property
    def bar(self):
        return 'hello'

foo = Foo()
foo

foo.bar

foo.bar = 'world'

