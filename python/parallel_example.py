##################################
#  Example 1 Parallel python
#################################
import pp
import math, sys, time

def sum_all(n):
    """Calculates sum of n"""
    return sum([x for x in xrange(1,n)])

#Setting the parallelization 

# tuple of all parallel python servers to connect with
ppservers = ()  #create a list of all nodes, can be ignored in single coputer

#edit the numbers of cpus here
ncpus = 4

#or by default:
#job_server = pp.Server()

job_server = pp.Server(ncpus, ppservers=ppservers)

print "Starting pp with", job_server.get_ncpus(), "workers"

#time
start_time = time.time()

# The following submits 10 jobs and then retrieves the results
#Numeros = (1000000, 1001000, 1000200, 1003000, 1004000, 1005000, 1006000, 1007000,10000,120000)
# Or in a for loop..  

Numeros = [n for n in range (10000000,10000010)]

print Numeros
# magic begins...

jobs = [(input, job_server.submit(sum_all,(input,),)) for input in Numeros]
for input, job in jobs:
    print "Sum of Numbers", input, "is", job()

print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()    

##################################
#  Example 2 Parallel python
#################################
import pp
import math, sys, time
import pp

#boundaries 
def isprime(n):
    """Returns True if n is prime and False otherwise"""
    if not isinstance(n, int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i += 1
    return True

def sum_primes(n):
    """Calculates sum of all primes below given integer n"""
    return sum([x for x in xrange(2,n) if isprime(x)])

start_time = time.time()

# tuple of all parallel python servers to connect with
ppservers = ()

#edit the numbers of cpus here
#ncpus = 1 
#job_server = pp.Server(ncpus, ppservers=ppservers)

#or autodetec the  N of cpus
job_server = pp.Server(ncpus='autodetect', ppservers=ppservers)


print "Starting pp with", job_server.get_ncpus(), "workers"


#Numeros = (1,0,1,100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700,10000,120000)
Numeros = ((int(n)) for n in range (1000000,1000010))

jobs = [(input, job_server.submit(sum_primes,(input,), (isprime,), ("math",))) for input in Numeros]
for input, job in jobs:
    print "Sum of primes below", input, "is", job()

print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()    




