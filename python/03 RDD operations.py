# All numbers from 0 to 1000. Split in 4 partitions
numbers = sc.parallelize( range(0,1001), 4 )

print numbers.getNumPartitions()
print numbers.count()
print numbers.take(10)

# Transformation: take only the odd numbers
odd = numbers.filter( lambda x : x % 2 )

odd.take(10)  # action

# Transformation: compute the cosine of each number
from math import cos
odd_cosine = odd.map( cos )

odd_cosine.take(10) # action

# Action: sum all values
from operator import add
result = odd_cosine.reduce( add )
print result

a = sc.parallelize( xrange(20), 4 )

b1 = a.map( lambda x : x*x )

from operator import add
result1 = b1.reduce( add )

print result1

b2 = a.flatMap( lambda x : x*x )

# This will trigger an error
b2.take(1)

# Ensure flatMap returns a list, even if it's a list of 1
b2 = a.flatMap( lambda x : [x*x] )

result2 = b2.reduce( add )

print result2
result2 == result1

b2b = a.flatMap( lambda x : [x, x*x] )

b2b.take(6)

# In Python, the easiest way of returning an iterator is by creating 
# a generator function via yield
def mapper( it ):
    for n in it:
        yield n*n

# Now we have the function, let's use it
b3 = a.mapPartitions( mapper )
result3 = b3.reduce( add )
print result3
result3 == result1

# In Python, the easiest way of returning an iterator is by creating 
# a generator function via yield
def mapper( partitionIndex, it ):
    for n in it:
        yield n*n

# Now we have the function, let's use it
b4 = a.mapPartitionsWithIndex( mapper )
result4 = b4.reduce( add )
print result4
result4 == result1

