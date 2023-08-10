from __future__ import print_function, division

from thinkbayes2 import Pmf

from random import random

def flip(p):
    return random() < p

def run_single_simulation(func, iters=1000000):
    pmf_t = Pmf([0.2, 0.4])
    p = 0.1
    s = 0.9

    outcomes = Pmf()
    post_t = Pmf()
    for i in range(iters):
        test, sick, t = func(p, s, pmf_t)
        if test:
            outcomes[sick] += 1
            post_t[t] += 1

    outcomes.Normalize()
    post_t.Normalize()
    return outcomes, post_t

def generate_patient_A(p, s, pmf_t):
    while True:
        t = pmf_t.Random()
        sick = flip(p)
        test = flip(s) if sick else flip(t)
        return test, sick, t
                
outcomes, post_t = run_single_simulation(generate_patient_A)
outcomes.Print()
post_t.Print()

def generate_patient_B(p, s, pmf_t):
    t = pmf_t.Random()
    while True:
        sick = flip(p)
        test = flip(s) if sick else flip(t)
        return test, sick, t
                
outcomes, post_t = run_single_simulation(generate_patient_B)
outcomes.Print()
post_t.Print()

def generate_patient_C(p, s, pmf_t):
    while True:
        t = pmf_t.Random()
        sick = flip(p)
        test = flip(s) if sick else flip(t)
        if test:
            return test, sick, t
                
outcomes, post_t = run_single_simulation(generate_patient_C)
outcomes.Print()
post_t.Print()

def generate_patient_D(p, s, pmf_t):
    t = pmf_t.Random()
    while True:
        sick = flip(p)
        test = flip(s) if sick else flip(t)
        if test:
            return test, sick, t
                
outcomes, post_t = run_single_simulation(generate_patient_D)
outcomes.Print()
post_t.Print()

from random import choice
import numpy as np 
N = 100
patients = range(N)

p = 0.1
s = 0.9
num_sick = 0

pmf_t = Pmf()
pmf_sick = Pmf()

for i in range(10000000):
    # decide what the value of t is
    t = choice([0.2, 0.4])
    np.random.shuffle(patients)
    
    # generate patients until we get a positive test
    for patient in patients:
        sick = flip(p)
        test = flip(s) if sick else flip(t)
        if test:
            if patient==1:
                #print(patient, sick, t)
                pmf_t[t] += 1
                pmf_sick[sick] += 1
            break
            
pmf_t.Normalize()
pmf_sick.Normalize()

print('Dist of t')
pmf_t.Print()
print('Dist of status')
pmf_sick.Print()

num_sick



