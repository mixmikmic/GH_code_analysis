#dependencies
import time
import math

def prime_generator( limit ):
    ls = [True] * limit
    
    for i in range(2, int(math.sqrt(limit)) + 1): #iterate over all possible factors
        for j in range(i*2, limit, i): # and their multiples
            ls[j] = False #cross 'em out
    
    print( "We  have ", ls.count(True) - 2, " prime numbers." )
    print( [i for i in range(2, limit) if ls[i]]) #get the rest

start = time.time()
prime_generator(10)
print("-----Simple: %s seconds -----" % (time.time() - start))

start = time.time()
prime_generator(100)
print("-----Simple: %s seconds -----" % (time.time() - start))

start = time.time()
prime_generator(1000)
print("-----Simple: %s seconds -----" % (time.time() - start))

