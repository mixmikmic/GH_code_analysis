from numpy import array
A=array([[60,5.5,1],[65,5.0,0],[55,6.0,1]])
b=array([[66,70,78]])
print(A)
print(b)

from numpy.linalg import inv
x=inv(A).dot(b.T)
print(x)

from numpy.linalg import solve
x = solve(A,b.T)
print(x)

