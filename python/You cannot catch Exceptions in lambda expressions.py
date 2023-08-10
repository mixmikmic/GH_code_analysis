def add_str(s):
    
    try:
        return sum([int(i) for i in s.split('+')])
    except AttributeError:
        return None

print(add_str(1+2))

l_add_str = lambda s: sum([int(i) for i in s.split('+')])

print(l_add_str(1+2))

