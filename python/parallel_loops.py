import numpy as np
import math
import joblib

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))

joblib.Parallel(n_jobs=2)(joblib.delayed(math.sqrt)(i ** 2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

joblib.Parallel(n_jobs=2, backend="threading")(joblib.delayed(math.sqrt)(i ** 2) for i in range(10))

with joblib.Parallel(n_jobs=2) as parallel:
    accumulator = 0.
    n_iter = 0
    while accumulator < 1000:
        results = parallel(joblib.delayed(math.sqrt)(accumulator + i ** 2)
                           for i in range(5))
        accumulator += sum(results)  # synchronization barrier
        n_iter += 1

from multiprocessing import Pool
from functools import partial

def f(a, b, c):
    print("{} {} {}".format(a, b, c))

def main():
    iterable = [1, 2, 3, 4, 5]
    pool = Pool()
    a = "hi"
    b = "there"
    func = partial(f, a, b)
    pool.map(func, iterable)
    pool.close()
    pool.join()

main()

map(lambda x : f(x,10,10), [1,3,4])

def funcPool(x, y, z):
    return x+y+z

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

# Make the Pool of workers
pool = ThreadPool(4) 
# Open the urls in their own threads
# and return the results
results = pool.map(lambda x:funcPool(x, 2, 2), [1,2,3,4,5])
print(results)
#close the pool and wait for the work to finish 
pool.close() 
pool.join() 



