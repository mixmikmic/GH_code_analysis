def square_list(in_list): 
    result = []
    for i in in_list: 
        result.append(i**2)
    return result

test_list = [1, 2, 3, 4, 5, 6]
square_list(test_list)

def gen_func_square(in_list): 
    for i in in_list: 
        #yield the next result 
        yield (i**2)
        
gen_func_square(test_list)

#create the generator object, assign to variable 
gen_object = gen_func_square(test_list)

#sequentially pass through generator object with next 
print (next(gen_object))
print (next(gen_object))
print (next(gen_object))
print (next(gen_object))
print (next(gen_object))
print (next(gen_object))

print (next(gen_object))

#need to recreate gen object since it was previously exhausted
gen_object = gen_func_square(test_list)

for square in gen_object: 
    print (square)

import time 
import random
import numpy as np
from pympler import asizeof #memory profiler 

#loop through length of how_long
#calculate mean of random vector 
#append that mean to a running list 

def mean_list(how_long):    
    means = []
    for i in range(how_long):
        #create vector of 100 random numbers range=[0, 100]
        vec = np.random.uniform(0, 100, size=100)
        avg = np.mean(vec)
        means.append(avg)
    return means 

#do the same with a generator 
def mean_gen(how_long):    
    for i in range(how_long):
        vec = np.random.uniform(0, 100, size=100)
        yield np.mean(vec)

#here I am just timing how long it takes to run the function 
start_time = time.clock()
test_avg = mean_list(1000000)
end_time = time.clock()

#and the memory it uses 
byte_size = asizeof.asizeof(test_avg)
MB = byte_size/1000000

print ('process took {} seconds'.format(end_time-start_time))
print ('process used {} megabytes'.format(MB))

#here I am just timing how long it takes to run the function 
start_time = time.clock()
test_avg = mean_gen(1000000)
end_time = time.clock()

#and the memory it uses 
byte_size = asizeof.asizeof(test_avg)
MB = byte_size/1000000.

print ('process took {} seconds'.format(end_time-start_time))
print ('process used {} bytes'.format(byte_size))

start_time = time.clock()
test_avg = mean_gen(1000000)
for i in test_avg: 
    pass
end_time = time.clock()
print ('process took {} seconds'.format(end_time-start_time))

mean_gen_toList = list(mean_gen(10000))
mean_gen_toList[5064]



