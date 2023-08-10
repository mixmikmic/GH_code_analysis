from collections import defaultdict
# define a default function
def mydefault():
    print("call mydefault")
    return list()


s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(mydefault)
for k, v in s:
    print("access: ", k)
    d[k].append(v)

sorted(d.items())



