# Create array
import numpy as np
myList = [4,5,6]
myArray = np.array(myList)
print myArray
print myArray.shape

# Access Data
myList = [[4,5,6],[6,7,8]]
myArray = np.array(myList)
print myArray
print myArray.shape
print 'First row = %s'%myArray[0]
print 'Last row = %s'%myArray[-1]
print 'Specific row and column = %s'%myArray[0,1]
print 'whole column = %s'%myArray[:,1]

# Arithmetic
myArray1 = np.array([1,1,1])
myArray2 = np.array([2,1,3])
print 'Addition = %s'%(myArray1 + myArray2)
print 'Multiplication = %s'%(myArray1 * myArray2)

# hstack Stack arrays in sequence horizontally (column wise)
alligatorLength = np.array([[3.87], [3.61], [4.33], [3.43], [3.81], [3.83], [3.46], [3.76], [3.50], [3.58], [4.19], [3.78], [3.71], [3.73], [3.78]])
alligatorWeight = np.array([[4.87], [3.93], [6.46], [3.33], [4.38], [4.70], [3.50], [4.50], [3.58], [3.64], [5.90], [4.43], [4.38], [4.42], [4.25]])
alligatorArray = np.hstack((alligatorLength, alligatorWeight))
alligatorArray

# vstack Stack arrays in sequence vertically (row wise)
alligatorLength = np.array([3.87, 3.61, 4.33, 3.43, 3.81, 3.83, 3.46, 3.76, 3.50, 3.58, 4.19, 3.78, 3.71, 3.73, 3.78])
alligatorWeight = np.array([4.87, 3.93, 6.46, 3.33, 4.38, 4.70, 3.50, 4.50, 3.58, 3.64, 5.90, 4.43, 4.38, 4.42, 4.25])
alligatorArray = np.vstack((alligatorLength, alligatorWeight))
alligatorArray

