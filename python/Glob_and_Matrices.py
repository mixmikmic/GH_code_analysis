import numpy as np
from glob import glob

# How to load a large number of files at once
fnames = glob('/home/mrice/Desktop/Astro120/UGSI_files/Lab2_tutorials/lab2_week1/incand120_data/incand120*')
incand_files = [np.genfromtxt(f, skiprows=17, skip_footer=1, usecols=(0,1)) for f in fnames]
print len(incand_files)

import numpy as np
# Initiate a matrix to work with
x = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])

# Example 1 of matrix dot product; works because matrices are both square.
y1 = x**2
print y1

# Example 2 of matrix dot product
y2 = np.dot(x,x)
print y2

# Non-square matrix
x2 = np.matrix([[1,2,3],[4,5,6]])
x2_transpose = np.transpose(x2)
print x2
print x2_transpose

# Matrix dot product still works for non-square matrices
print np.dot(x2_transpose, x2)

# Now, let's invert a matrix
n = np.array([[1.,2.],[3.,4.]])

# Inverse of n
x3 = np.linalg.inv(n)
print x3

# Dot product between the two; produces identity matrix
x3_dot_n = np.dot(x3, n)
print x3_dot_n



