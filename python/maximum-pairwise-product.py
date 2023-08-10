get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -u -d -v")

import random

random.seed(1)

a = [7, 1, 2, 8, 1, 3]
rnd_lst = random.sample(range(0, 10000000), 1000)

import sys

def max_pairprod_1(ary):
    n = len(ary)
    max_prod = -sys.float_info.max
    
    for i in range(0, n):
        for j in range(i + 1, n):
            prod = ary[i] * ary[j]
            if prod > max_prod:
                max_prod = prod
                
    return max_prod

print(max_pairprod_1(ary=a))
print(max_pairprod_1(ary=rnd_lst))

import sys

def max_pairprod_2(ary):
    pos_1 = -sys.float_info.max
    pos_2 = -sys.float_info.max
    
    for i in range(0, len(ary)):
        if ary[i] > pos_1:
            tmp = pos_1
            pos_1 = ary[i]
            if tmp > pos_2:
                pos_2 = tmp
        elif ary[i] > pos_2:
            pos_2 = ary[i]

    return pos_1 * pos_2

print(max_pairprod_2(ary=a))
print(max_pairprod_2(ary=rnd_lst))

get_ipython().magic('timeit -n 1000 -r 3 max_pairprod_1(ary=rnd_lst)')

get_ipython().magic('timeit -n 1000 -r 3 max_pairprod_2(ary=rnd_lst)')

import timeit

funcs = ['max_pairprod_1', 'max_pairprod_2']
orders_n = [10**n for n in range(1, 5)]
times_n = {f:[] for f in funcs}

for n in orders_n:
    rnd_lst = random.sample(range(0, 10**6), n)
    for f in funcs:
        times_n[f].append(min(timeit.Timer('%s(rnd_lst)' % f, 
                'from __main__ import %s, rnd_lst' % f)
                    .repeat(repeat=3, number=5)))

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def plot_timing():

    labels = [('max_pairprod_1', 'max_pairprod_1'), 
              ('max_pairprod_2', 'max_pairprod_2')]

    plt.rcParams.update({'font.size': 12})

    fig = plt.figure(figsize=(10, 8))
    for lb in labels:
        plt.plot(orders_n, times_n[lb[0]], 
             alpha=0.5, label=lb[1], marker='o', lw=3)
    plt.xlabel('sample size n')
    plt.ylabel('time per computation in milliseconds [ms]')
    plt.legend(loc=2)
    plt.ylim([-1, 60])
    plt.grid()
    plt.show()

plot_timing()

