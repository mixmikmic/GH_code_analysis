import numpy as np

#Array definitions
v=np.array([1.4,1,5,1.6]) #implict float typing
print("Float array:",v)

v=np.array([1,2,3,4]) #implicit int typing
print("Int array:",v)


v=np.array([1,2,3,4],dtype=np.float) #explicit float typing
print("Explicit float array:",v)


#Make a random array
v = np.random.random(10)
print("Random array contents:",v)

#Sorting
v.sort()
print("Sorted random array:",v)

#1. Python arrays are flexible
python_v=[1,2,3,4,5.]
print("Mixed type array:", python_v)

#2. Numpy arrays are fixed variable type, but much more compact so they are faster!
from timeit import Timer
from sys import getsizeof

NTestTimes = 100 #

list_v=range(100000)
np_v=np.arange(100000)

t_numpy = Timer("np_v.sum()", "from __main__ import np_v")
t_list = Timer("sum(list_v)", "from __main__ import list_v")
print("Numpy average time to sum: %.3e" % (t_numpy.timeit(NTestTimes)/NTestTimes,))
print("List average time to sum:  %.3e" % (t_list.timeit(NTestTimes)/NTestTimes,))

print("Memory size of numpy array: %d bytes"% np_v.nbytes)
print("Memory size of python list: %d bytes"% (getsizeof(list_v) + sum(getsizeof(i) for i in list_v)))

#3. Matrix operations
theta=np.radians(45)
c, s = np.cos(theta), np.sin(theta)
R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
v=np.random.random(3)

print("Random vector:",v)
print("Random vector rotated 45 degrees in yz plane:",R.dot(v))

