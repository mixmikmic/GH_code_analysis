from contextlib import contextmanager
from traceback import print_exc, format_exc

@contextmanager
def tag(name):
    print "<{}>".format(name)
    yield "hello"
    print "</{}>".format(name)

with tag("h1") as t:
    print t + " world"
    

class MyCtxMgr(object):
    def __init__(self):
        pass
    def __enter__(self):
        print "entering"
        return self
    def __exit__(self, exc, exc2, traceback):
        print "exiting"
        print exc
        print exc2
        print traceback
        return True

with MyCtxMgr() as m:
    print "in the middle"
    raise ValueError("Bad egg")

    

class MyGetSet(object):
    def __init__(self, val):
        print 'created val : %s' % val
        self._val = val
        
    @property
    def val(self):
        print 'getting val'
        return self._val
    
    @val.setter
    def val(self, v):
        print 'setting val to %s' % v
        self._val = v
        
    @val.deleter
    def val(self):
        print 'deleting val : %s' % self._val
        del self._val
        
obj = MyGetSet(20)
print obj.val
obj.val = 25
print obj.val
del obj.val
print getattr(obj, 'val', 'Val no longer exists')



from math import sqrt

print 1.0/sqrt(3)


def tag(tg, br=True, data=""):
    if br or isinstance(data, (list, tuple)):
        result = ["<%s>\n" % tg]
        if isinstance(data, (list, tuple)):
            for i, line in enumerate(data):
                data[i] = "  " + line
            result.extend(data)
        else:
            result.append("  %s\n" % data)
        result.append("</%s>\n" % tg)
    else:
        result = ["<%s>%s</%s>\n" % (tg, data, tg)]
    return result
    
res = tag("html", data =
        tag("head", data =
          tag("style", data =
                "body { font-size: 10px; }"
          )
        ) +
        tag("body", data =
          tag("div", data =
                "Hello"
          )
        )
      )

print "".join(res)
    
    
    
    



