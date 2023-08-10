import numpy as np

n = 20

while True:
    res = [n/x for x in range(1,21)]
    res = [r.is_integer() for r in res]
    if np.all(res):
        print(n)
        break
    else:
        n += 20

def func():
    i = 1
    j = 2520
    while i < 21:
        if j % i != 0:
            j += 20
            i = 1
        else:
            i += 1
    return j



for i in range(4):
    print(i)

for i in range(4):
    print(i)
    i = 1



