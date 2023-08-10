from itertools import product

def combinations(vals):
    B = []
    for w in product(vals, repeat=len(vals)):
        B.append(w)
    return B

def display_group(vals):
    print("===")
    for i in vals:
        print(i)

        
def remove_dups(vals):
    ind = 0
    i = vals[ind]
    while ind < len(vals):
        if (vals.count(i) > 1):
            vals.remove(i)
            ind -= 1
        ind += 1
        if (ind >= len(vals)):
            break
        i = vals[ind]

import pandas
L = "qaw"
B = combinations(L)
#df = pandas.DataFrame(B)
#df.head()
remove_dups(B)
display_group(B)

