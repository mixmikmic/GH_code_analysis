get_ipython().magic('matplotlib notebook')
import numpy
from matplotlib import pyplot

def matmult1(A, x):
    """Entries of y are dot products of rows of A with x"""
    y = numpy.zeros_like(A[:,0])
    for i in range(len(A)):
        row = A[i,:]
        for j in range(len(row)):
            y[i] += row[j] * x[j]
    return y

A = numpy.array([[1,2],[3,5],[7,11]])
x = numpy.array([10,20])
matmult1(A, x)

def matmult2(A, x):
    """Same idea, but more compactly"""
    y = numpy.zeros_like(A[:,0])
    for i,row in enumerate(A):
        y[i] = row.dot(x)
    return y

matmult2(A, x)

def matmult3(A, x):
    """y is a linear expansion of the columns of A"""
    y = numpy.zeros_like(A[:,0])
    for j,col in enumerate(A.T):
        y += col * x[j]
    return y

matmult3(A, x)

# We will use this version
A.dot(x)

B = numpy.array([[2, 3],[0, 4]])
print(B)
print(B.dot(B.T), B.T.dot(B))
Binv = numpy.linalg.inv(B)
Binv.dot(B), B.dot(Binv)

x = numpy.linspace(-1,1)
A = numpy.array([x**3, x**2, x, 1+0*x]).T
print('shape =', A.shape)  # This is a tall matrix with 4 columns
pyplot.style.use('ggplot')
pyplot.figure()
pyplot.plot(x, A)
pyplot.ylim((-1.1,1.1))
pyplot.show()

pyplot.figure()
p = numpy.array([5,0,-3,0])
pyplot.plot(x, A.dot(p))

x1 = numpy.array([-0.9, 0.1, 0.5, 0.8])  # points where we know values
y = numpy.array([1, 2.4, -0.2, 1.3])     # values at those points
pyplot.figure()
pyplot.plot(x1, y, '*')
B = numpy.vander(x1)                     # Vandermonde matrix at the known points
p = numpy.linalg.solve(B, y)             # Compute the polynomial coefficients
print(p)
pyplot.plot(x, A.dot(p))                 # Plot the polynomial evaluated at all points
print('B =', B, '\np =', p)

# Make some polynomials
q0 = A.dot(numpy.array([0,0,0,.5]))  # .5
q1 = A.dot(numpy.array([0,0,1,0]))  # x
q2 = A.dot(numpy.array([0,1,0,0]))  # x^2

pyplot.figure()
pyplot.plot(x, numpy.array([q0, q1, q2]).T)

# Inner products of even and odd functions

q0 = q0 / numpy.linalg.norm(q0)
q1.dot(q0), q2.dot(q0), q2.dot(q1)

# What is the constant component of q2?

pyplot.figure()
pyplot.plot(x, q2.dot(q0)*q0)

# Let's project that away so that q2 is orthogonal to q0

q2 = q2 - q2.dot(q0)*q0

Q = numpy.array([q0, q1, q2]).T
print(Q.T.dot(Q))
pyplot.figure()
pyplot.plot(x, Q)

def gram_schmidt_naive(X):
    Q = numpy.zeros_like(X)
    R = numpy.zeros((len(X.T),len(X.T)))
    for i in range(len(Q.T)):
        v = X[:,i].copy()
        for j in range(i):
            r = v.dot(Q[:,j])
            R[j,i] = r
            v -= r * Q[:,j]    # "modified Gram-Schmidt" - remove each component before next dot product
        R[i,i] = numpy.linalg.norm(v)
        Q[:,i] = v / R[i,i]
    return Q, R

Q, R = gram_schmidt_naive(A)
print(Q.T.dot(Q))
print(numpy.linalg.norm(Q.dot(R)-A))
pyplot.figure()
pyplot.plot(x, Q)

Q, R = gram_schmidt_naive(numpy.vander(x, 4, increasing=True))
pyplot.figure()
pyplot.plot(x, Q)

x1 = numpy.array([-0.9, 0.1, 0.5, 0.8])  # points where we know values
y = numpy.array([1, 2.4, -0.2, 1.3])     # values at those points
pyplot.figure()
pyplot.plot(x1, y, '*')
B = numpy.vander(x1, 2)                     # Vandermonde matrix at the known points
Q, R = gram_schmidt_naive(B)
p = numpy.linalg.solve(R, Q.T.dot(y))             # Compute the polynomial coefficients
print(p)
pyplot.plot(x, numpy.vander(x,2).dot(p))                 # Plot the polynomial evaluated at all points
print('B =', B, '\np =', p)

m = 20
V = numpy.vander(numpy.linspace(-1,1,m), increasing=False)
Q, R = gram_schmidt_naive(V)

def qr_test(qr, V):
    Q, R = qr(V)
    m = len(Q.T)
    print(qr.__name__, numpy.linalg.norm(Q.dot(R) - V), numpy.linalg.norm(Q.T.dot(Q) - numpy.eye(m)))
    
qr_test(gram_schmidt_naive, V)
qr_test(numpy.linalg.qr, V)

def gram_schmidt_classical(X):
    Q = numpy.zeros_like(X)
    R = numpy.zeros((len(X.T),len(X.T)))
    for i in range(len(Q.T)):
        v = X[:,i].copy()
        R[:i,i] = Q[:,:i].T.dot(v)
        v -= Q[:,:i].dot(R[:i,i])
        R[i,i] = numpy.linalg.norm(v)
        Q[:,i] = v / R[i,i]
    return Q, R

qr_test(gram_schmidt_classical, V)

def gram_schmidt_modified(X):
    Q = X.copy()
    R = numpy.zeros((len(X.T), len(X.T)))
    for i in range(len(Q.T)):
        R[i,i] = numpy.linalg.norm(Q[:,i])
        Q[:,i] /= R[i,i]
        R[i,i+1:] = Q[:,i+1:].T.dot(Q[:,i])
        Q[:,i+1:] -= numpy.outer(Q[:,i], R[i,i+1:])
    return Q, R

qr_test(gram_schmidt_modified, V)

def householder_Q_times(V, x):
    """Apply orthogonal matrix represented as list of Householder reflectors"""
    y = x.copy()
    for i in reversed(range(len(V))):
        y[i:] -= 2 * V[i] * V[i].dot(y[i:])
    return y

def qr_householder1(A):
    "Compute QR factorization using naive Householder reflection"
    m, n = A.shape
    R = A.copy()
    V = []
    for i in range(n):
        x = R[i:,i]
        v = -x
        v[0] += numpy.linalg.norm(x)
        v = v/numpy.linalg.norm(v)     # Normalized reflector plane
        R[i:,i:] -= 2 * numpy.outer(v, v.dot(R[i:,i:]))
        V.append(v)                    # Storing reflectors is equivalent to storing orthogonal matrix
    Q = numpy.eye(m, n)
    for i in range(n):
        Q[:,i] = householder_Q_times(V, Q[:,i])
    return Q, numpy.triu(R[:n,:])

qr_test(qr_householder1, numpy.array([[1.,2],[3,4],[5,6]]))

qr_test(qr_householder1, V)
qr_test(numpy.linalg.qr, V)

qr_test(qr_householder1, numpy.eye(1))

qr_test(qr_householder1, numpy.eye(3,2))

qr_test(qr_householder1, numpy.array([[1.,1], [2e-8,1]]))
print(qr_householder1(numpy.array([[1.,1], [2e-8,1]])))

def qr_householder2(A):
    "Compute QR factorization using Householder reflection"
    m, n = A.shape
    R = A.copy()
    V = []
    for i in range(n):
        v = R[i:,i].copy()
        v[0] += numpy.sign(v[0])*numpy.linalg.norm(v) # Choose the further of the two reflections
        v = v/numpy.linalg.norm(v)     # Normalized reflector plane
        R[i:,i:] -= 2 * numpy.outer(v, v.dot(R[i:,i:]))
        V.append(v)                    # Storing reflectors is equivalent to storing orthogonal matrix
    Q = numpy.eye(m, n)
    for i in range(n):
        Q[:,i] = householder_Q_times(V, Q[:,i])
    return Q, numpy.triu(R[:n,:])

qr_test(qr_householder2, numpy.eye(3,2))
qr_test(qr_householder2, numpy.array([[1.,1], [1e-8,1]]))
print(qr_householder2(numpy.array([[1.,1], [1e-8,1]])))

qr_test(qr_householder2, V)

def R_solve(R, b):
    """Solve Rx = b using back substitution."""
    x = b.copy()
    m = len(b)
    for i in reversed(range(m)):
        x[i] -= R[i,i+1:].dot(x[i+1:])
        x[i] /= R[i,i]
    return x

Q, R = numpy.linalg.qr(A)
b = Q.T.dot(A.dot(numpy.array([1,2,3,4])))
numpy.linalg.norm(R_solve(R, b) - numpy.linalg.solve(R, b))
R_solve(R, b)

# Test accuracy of solver for an ill-conditioned square matrix

x = numpy.linspace(-1,1,19)
A = numpy.vander(x)
print('cond(A) = ',numpy.linalg.cond(A))
Q, R = numpy.linalg.qr(A)
print('cond(R^{-1} Q^T A) =', numpy.linalg.cond(numpy.linalg.solve(R, Q.T.dot(A))))
L = numpy.linalg.cholesky(A.T.dot(A))
print('cond(L^{-T} L^{-1} A^T A) =', numpy.linalg.cond(numpy.linalg.solve(L.T, numpy.linalg.solve(L, A.T.dot(A)))))



m = 10
x = numpy.cos(numpy.linspace(0,numpy.pi,m))
f = 1.0*(x > 0) + (x < 0.5)
A = numpy.vander(x)
Q, R = numpy.linalg.qr(A)
p = numpy.linalg.solve(R, Q.T.dot(f))
y = numpy.linspace(-1,1,50)
g = numpy.vander(y, m).dot(p)
pyplot.figure()
pyplot.plot(x, f, '*')
pyplot.plot(y, g)
print(numpy.linalg.cond(A))

'%10e' % numpy.linalg.cond(numpy.vander(numpy.linspace(-1,1,100),20))



