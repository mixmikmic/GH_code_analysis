import numpy as np

# List to np array
arr = [1,2,3]
np_arr = np.array(arr)

type(arr),type(np_arr)

# Create Learge array in a range
# Semillar to python range function
arr = np.arange(0,10,1) # Start, end, step
print(arr)

# Zero
# 1d
print(np.zeros(4))
print('='*22)
# Multi dimentional
print(np.zeros((4,5)))

# Same for np.ones()

# LineSpace
np.linspace(0,11,10) # start, end , number of output

# Random Number
print(np.random.randint(0,100))

# for dimention 
np.random.randint(0,100,(3,3))

"""
Random seed helps to create some naumber everytime you ran the code
"""

for i in range(4):
    np.random.seed(1)
    print(np.random.randint(0,100,10))
print('WA seed')
print(np.random.randint(0,100,10))

arr = np.random.randint(0,100,10)
arr

# Reshape
arr.reshape(2,5)

arr = np.random.randint(0,100, (4,5))
arr

arr[1,2]

arr[1:,2:]

