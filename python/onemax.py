from PTO import random, solve

fitness = sum

def randsol():
    return [random.choice([0,1]) for x in range(10)]

for i in range(5):
    x = randsol()
    print("Random solution: fitness %d; %s" % (fitness(x), str(x)))

ind, fit = solve(randsol, fitness, solver="EA")
print(fit, ind)

ind, fit = solve(randsol, fitness, solver="HC", budget=15)
print(fit, ind)
ind, fit = solve(randsol, fitness, solver="HC", budget=150)
print(fit, ind)
ind, fit = solve(randsol, fitness, solver="HC", effort=1)
print(fit, ind)
ind, fit = solve(randsol, fitness, solver="HC", effort=2)
print(fit, ind)



