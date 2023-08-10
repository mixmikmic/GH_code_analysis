get_ipython().magic('pylab inline')

L = [1.32, 5, 'seven']
for i in range(len(L)):
    print "item %s of L is %s of type %s"  % (i, L[i], type(L[i]))

x = array([1.32, 5, 7])
for i in range(len(x)):
    print "item %s of x is %s of type %s"  % (i, x[i], type(x[i]))

x = array([1., 2, 3])
print 'x = ',x
print 'x has shape', x.shape

y = x.T  # transpose
print 'y = ',y
print 'y has shape', y.shape

z = linspace(0,2,5)
z

A = array([[1., 2, 3], [4,5,6]])
print 'A = \n', A  # \n means new line

u = array([[1., 2, 3]])
print 'u = ', u

v = u.T  
print '\nv = \n', v

print dot(A, v)

print dot(A.T, A)

print "A =\n", A  # \n means new line
print "\nA[:, 0:2] =\n", A[:,0:2]
print "\nA[:, 1:] =\n", A[:,1:]
print "\nA[:, :-1] =\n", A[:,:-1]
print "\nA[:, 2] =\n", A[:,2]
print "\nA[:, 2:3] =\n", A[:,2:3]

m,n = shape(A)  # number of rows and columns of A
print "Each column of A has %s elements, so product y=Av will too"  % m

y = zeros((m,1))  # initialize to zero-vector of correct shape (2D array with shape m by 1)
for j in range(n):
    # loop over columns and add in v[j] times j'th column of A:
    y += A[:,j:j+1]*v[j]   # use j:j+1 to select only j'th column as vector

print "\ny =\n", y

from numpy import linalg

print "The rank of A is ", linalg.matrix_rank(A)

B = dot(A, A.T)
print "B =\n", B

Binv = linalg.inv(B)
print "\nThe inverse of B is \n", Binv

print "\nThe product of the two is\n", dot(Binv, B)

xtrue = array([[3.], [-2.]])
print "For comparison, the true solution will be xtrue =\n", xtrue

# compute right hand side:
y = dot(B, xtrue)
print "\nFrom this we generate the right hand side y = B*xtrue = \n", y
print "\nMultiplying Binv * y gives\n", dot(Binv,y)

# Solve system:
x = linalg.solve(B, y)
print "\nSolving system directly gives\n", x



