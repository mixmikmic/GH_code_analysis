import numpy as np
import pandas as pd
import random

def target(sample):
    alfa,batch,l1,l2,l3,l4 = sample.values()
    #dummy target function
    return (alfa*batch*(l1-l2**2-l3+l4))**2

def create_initials(num):
    #creates initial population
    pop = []
    for i in range(num):
        node = {'a' : random.choice(a),'b' : random.choice(b), 'l1' : random.choice(l)}
        node['l2'] =  random.randint(0,node['l1'])
        node['l3'] =  random.randint(0,node['l2'])
        node['l4'] = random.randint(0,node['l3'])
        pop.append(node)
    return pop

def cross(n1,n2):
    #pick two random indexes - keep, switch and avg
    k1,k2 = random.choice(n1.keys()), random.choice(n1.keys())
    n3,n4 = n1, n1
    n5,n6 = n2, n2
    n3[k1], n3[k2] = n2[k1] ,n2[k2] #switch
    n4[k1], n4[k2] = n1[k1] ,n1[k2]
    n5[k1], n5[k2] = (n1[k1]+n2[k1])/2 ,(n1[k2]+n2[k2])/2 #avg
    n6[k1], n6[k2] =  (n1[k1]+n2[k1])/2 ,(n1[k2]+n2[k2])/2 #avg
    return [n1,n2,n3,n4,n5,n6]
    

def pickpair(vals,used):
    totval = sum(values)
    play = random.random()*totval
    prev = 0
    nextv = 0
    for v in values:
        nextv+=v
        if play<=nextv and play>prev:
            p1 = values.index(v) # pickfirst
            break
            while p1 in used:
                p1+=1
    p2 = p1
    while p2 == p1:
        p2 = random.randint(0,len(values)-1)
    return p1,p2 # p1 - chosen gradually, p2 - randomly
            
    

def nextgen(values,pop, size):
    # calculates next generation
    sval = sorted(values, reverse=True)
    newpop = []
    for v in sval[size/2]:
        newpop.append(pop[values.index(v)])
    while len(newpop<size):
        cand = random.choice(pop)
        if cand not in newpop:
            newpop.append(cand)
    return newpop

def mutation(pop):
    n  = random.randint(0,len(pop)-1)
    v = random.choice(pop[0].keys())
    if v == 'a':
        r = random.choice(a)
    elif v=='b':
        r = random.choice(b)
    else:
        r = random.choice(l)
    pop[n][v]=r
    return pop

# initial population ranges:


a = [100,10,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0001]
b = [50,100,200,500,1000,2000,5000]
l = [0,2,5,10,20,30,40,50]

size = 10000
bestval = 0
best = {}
iterations = 100
pairs = 100

random.seed(101)

pop = create_initials(size)

for i in range(iterations):
    values = []
    for s in pop:
        values.append(target(s))

    if max(values)>bestval:
        bestval = max(values)
        best = pop[values.index(bestval)]
        
    newpop = pop[:]
    used = []
    for p in range(pairs):
        pair = pickpair(values,used)
        used += list(pair)
        new = cross(pop[pair[0]],pop[pair[1]])
        newpop+=new
    
    pop = mutation(newpop)
        
    if i%(iterations/10)==0:print 'iteration:',i,'best:',bestval,best
        



