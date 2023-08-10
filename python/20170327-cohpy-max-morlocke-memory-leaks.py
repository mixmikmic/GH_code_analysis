def append_to(element, target=[]):
    target.append(element)
    return target

a = append_to('hello')
print(a)

b = append_to('world')
print(b)

a

a is b

def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target

c = append_to('hello')
print(c)

d = append_to('world')
print(d)

c

