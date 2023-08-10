import re

class Foo(object):
    @property
    def bar(self):
        return self.x
    
    @bar.setter
    def bar(self, x):
        self.x = 3 * x

foo = Foo()
foo

foo.bar = 'world'

foo.bar

goo = Foo()
goo

goo.bar

goo.foo

class Hoo(object):
    @property
    def bar(self):
        try:
            return self.x
        except AttributeError as e:
            # print('old e.args', repr(e.args))
            e.args = (re.sub(
                r"'[A-Za-z_][A-Za-z0-9_]*'$",
                "'bar'",
                e.args[0]), )
            e.args = (
                "{class_name!r} "
                "object has no attribute "
                "{attribute_name!r}".format(
                    class_name=self.__class__.__name__,
                    attribute_name='bar')
                , )
            raise e

    @bar.setter
    def bar(self, x):
        self.x = 3 * x

hoo = Hoo()
hoo

hoo.bar

