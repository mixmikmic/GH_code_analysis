from __future__ import print_function
import numpy as np
def get_sin(arr):
    # Create an empty output array of same size as input
    output = np.empty_like(arr)
    for i in range(len(output)):
        output[i] = np.sin(arr[i])
    return output

input_arr = np.random.uniform(-np.pi, np.pi, 10000000)
get_ipython().magic('time get_sin(input_arr)')

input_arr = np.random.uniform(-np.pi, np.pi, 10000000)
get_ipython().magic('time np.sin(input_arr)')

arr = np.random.randint(1, 100, (3, 4))
# take reciprocal
print("Original Array: \n{}".format(arr), end="\n\n")
print("Reciprocal: \n{}".format(1/arr), end="\n\n")

x = np.arange(-5, 5)
print("x      =", x)
print("x + 5  =", x + 10) # wrapper for np.sum 
print("x - 5  =", x - 10) # wrapper for np.subtract
print("x * 2  =", x * 4)  # wrapper for np.multiply
print("x / 2  =", x / 4)  # wrapper for np.divide
print("x % 2  =", x % 4)  # wrapper for np.mod
print("x // 2 =", x // 4) # wrapper for np.floor_divide
print("x ** 2 =", x ** 2) # wrapper for np.power
print("abs(x) =", abs(x)) # wrapper for np.abs

arr1 = np.array([1., 2., 3., 4.])
arr2 = np.linspace(4, 16, num=4)
print("Array1: \n{}".format(arr1), end="\n\n")
print("Array2: \n{}".format(arr2), end="\n\n")
print("\n Array2 - Array1: \n {}".format(arr2-arr1), end="\n\n")

arr2 = np.linspace(4, 16, num=3)
print("\n Array2 - Array1: \n {}".format(arr2-arr1), end="\n\n")

input_arr = np.random.uniform(-1, 1, 5)
print("Input Array: \n{}".format(input_arr), end="\n\n")
print("sin: \n{}".format(np.sin(input_arr)), end="\n\n")
print("cos: \n{}".format(np.cos(input_arr)), end="\n\n")
print("tan: \n{}".format(np.tan(input_arr)), end="\n\n")
print("arcsin: \n{}".format(np.arcsin(input_arr)), end="\n\n")
print("arccos: \n{}".format(np.arccos(input_arr)), end="\n\n")
print("arctan: \n{}".format(np.arctan(input_arr)), end="\n\n")

input_arr = np.random.randint(1, 7, 5)
print("x        =", input_arr)
print("ln(x)    =", np.log(input_arr))
print("log2(x)  =", np.log2(input_arr))
print("log10(x) =", np.log10(input_arr))

input_arr = np.random.randint(1, 7, 5)
print("x     =", input_arr)
print("e^x   =", np.exp(input_arr))
print("2^x   =", np.exp2(input_arr))
print("10^x   =", np.power(10, input_arr))

