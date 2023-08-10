from random import uniform

uniform(-1,1) 

x = uniform(-1,1)
y = uniform(-1,1)
if x**2 + y**2 < 1:
    print("Inside the circle")
else:
    print("Outside the circle")

for i in range(10):
    x = uniform(-1,1)
    y = uniform(-1,1)
    if x**2 + y**2 < 1:
        print("Inside the circle")
    else:
        print("Outside the circle")

trials = 1000
hits = 0
for i in range(trials):
    x = uniform(-1,1)
    y = uniform(-1,1)
    if x**2 + y**2 < 1:
        hits += 1
hits / trials

def estimate_pi(trials):
    hits = 0
    for i in range(trials):
        x = uniform(-1,1)
        y = uniform(-1,1)
        if x**2 + y**2 < 1:
            hits += 1
    return 4.0 * hits / trials

estimate_pi(1000)

for trials in [10000, 50000, 100000, 500000, 1000000]:
    print("With", trials," trials, the estimated pi is", estimate_pi(trials))

