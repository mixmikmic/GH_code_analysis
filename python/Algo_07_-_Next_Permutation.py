#dependencies
import time

def _next( current ):
    n = len(current) #len
    for i in reversed(range(n-1)):
        if current[i] < current[i+1]: #get pivot value if sequence breaks
            print ("Pivot: ", current[i])
            break #here the pivot is at i
    else: #last permutation
        return current
    for j in reversed(range(i, n)):
        if current[i] < current[j]:
            print ("Value to swap: ", current [j])
            current[i], current[j] = current[j], current[i]
            current[i + 1:] = reversed(current[i + 1:])
    return current

start = time.time()
print(_next([3,4,5,2,1]))
print("----- %s seconds -----" % (time.time() - start))

start = time.time()
print(_next([5, 4, 3, 2, 1]))
print("----- %s seconds -----" % (time.time() - start))

start = time.time()
print(_next( list('ACBD') ))
print("----- %s seconds -----" % (time.time() - start))

