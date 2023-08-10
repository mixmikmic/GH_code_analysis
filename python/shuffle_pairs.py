import numpy as np

xs = np.arange(10, 14)
ys = np.arange(20, 25)
print(xs, ys)

n = len(xs)
m = len(ys)
indices = np.arange(n)

array = np.tile(ys, (n, 1))
print(array)

[np.random.shuffle(array[i]) for i in range(n)] 
print(array)

counts = np.full_like(xs, m)
print(counts)

weights = np.array(counts, dtype=float)
weights /= np.sum(weights)
print(weights)

i = np.random.choice(indices, p=weights)
print(i)

counts[i] -= 1
pair = xs[i], array[i, counts[i]]
array[i, counts[i]] = -1
print(pair)

print(counts)

print(array)

weights = np.array(counts, dtype=float)
weights[i] = 0
weights /= np.sum(weights)
print(weights)

i = np.random.choice(indices, p=weights)
counts[i] -= 1
pair = xs[i], array[i, counts[i]]
array[i, counts[i]] = -1
print(pair)

print(counts)

print(array)

def generate_pairs(xs, ys):
    n = len(xs)
    m = len(ys)
    indices = np.arange(n)
    
    array = np.tile(ys, (n, 1))
    [np.random.shuffle(array[i]) for i in range(n)]
    
    counts = np.full_like(xs, m)
    i = -1

    for _ in range(n * m):
        weights = np.array(counts, dtype=float)
        if i != -1:
            weights[i] = 0
        weights /= np.sum(weights)

        i = np.random.choice(indices, p=weights)
        counts[i] -= 1
        pair = xs[i], array[i, counts[i]]
        array[i, counts[i]] = -1
        
        yield pair

for pairs in generate_pairs(xs, ys):
    print(pairs)



