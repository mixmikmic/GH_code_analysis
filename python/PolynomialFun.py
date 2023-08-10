get_ipython().magic('matplotlib inline')
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# lambda functions for each of the basis functions
p0 = lambda x: np.ones(x.shape)
p1 = lambda x: x
p2 = lambda x: x**2


# lambda function for the matrix whose columns are p_i(x)
A = lambda x: np.array([ p0(x), p1(x), p2(x)]).transpose()

# lambda function for any vector in P_2,  v = c[0]*p0 + c[1]*p1 + c[2]*p2
v = lambda c,x :  np.dot(A(x),c)

x = np.array([-1.,0.,1.])
print p0(x),p1(x),p2(x)

print A(x)

c = np.array([1,2,-1])
print v(c,x)


x = np.linspace(-1,1)
plt.figure()
plt.plot(x,p0(x),label='$p_0$')
plt.hold(True)
plt.plot(x,p1(x),label='$p_1$')
plt.plot(x,p2(x),label='$p_2$')
plt.xlabel('x')
plt.ylim(-1.5,1.5)
plt.legend(loc='best')
plt.grid()
plt.show()

x = np.array([-1.,0.,1.])
f = np.array([0.,2.,-1.])
c = la.solve(A(x),f)

# and plot it out
xx = np.linspace(-1,1) # use well sampled space for plotting the quadratic
plt.figure()
# plot the parabola
plt.plot(xx,v(c,xx),'r-')
# plot the interpolating points
plt.plot(x,f,'bo')
plt.xlabel('x')
plt.ylabel('$f(x)$')
plt.ylim(-1.5,2.5)
plt.title('$c={}$: $v ={}p_0 + {}p_1 + {}p_2$'.format(c,c[0],c[1],c[2]))
plt.grid()
plt.show()

# choose 7 evenly spaced points in [-1,1]
x = np.linspace(-1,1,7)

# perturb the parabola with uniform random noise
f = v(c,x) + np.random.uniform(-.5,.5,len(x))

# and plot with respect to the underlying parabola
plt.figure()
plt.plot(x,f,'bo')
plt.hold(True)
plt.plot(xx,v(c,xx),'r',label='v')
plt.xlabel('x')
plt.ylim(-1.5,2.5)
plt.grid()

# now calculate and plot the leastsquares solution to Ac = f
c_ls,res,rank,s = la.lstsq(A(x),f)

plt.plot(xx,v(c_ls,xx),'g',label='v_lstsq')
plt.title('$c={}$: $v={}p_0 + {}p_1 + {}p_2$'.format(c_ls,c_ls[0],c_ls[1],c_ls[2]))
plt.legend(loc='best')
plt.show()

# and show that this is the same solution we would get if we tried to solve the normal equations direction

AtA = np.dot(A(x).transpose(),A(x))
Atf = np.dot(A(x).transpose(),f)

c_norm = la.solve(AtA,Atf)

print 'numpy least-squares c = {}'.format(c_ls)
print 'normal equations      = {}'.format(c_norm)
print 'difference            = {}'.format(c_ls-c_norm)

print
print 'ATA ={}'.format(AtA)

# calculate the error vector
e = f - v(c_ls,x)

print 'error vector\n e={}\n'.format(e)

# and calculate the matrix vector product A^T e
print 'A^T e = {}'.format(np.dot(A(x).transpose(),e))

#  set the function to be projected
f = lambda x : np.cos(2*x) + np.sin(1.5*x)

# calculate the interpolation of f onto P2, when sampled at points -1,0,1
x = np.array([-1., 0., 1.])
c_interp = la.solve(A(x),f(x))


from scipy.integrate import quad

def mij(i,j,x):
    """ integrand for component Mij of the mass matrix"""
    p = np.array([1., x, x**2])
    return p[i]*p[j]

def fi(i,x,f):
    """ integrand for component i of the load vector"""
    p = np.array([1., x, x**2])
    return p[i]*f(x)





# construct the symmetric mass matrix  M_ij = <p_i,p_j> 

M = np.zeros((3,3))
fhat = np.zeros(3)
R = np.zeros((3,3)) # quadrature residuals

# loop over the upper triangular elements of M (and fill in the symmetric parts)
for i in range(0,3):
    fhat[i] = quad(lambda x: fi(i,x,f),-1.,1.)[0]
    for j in range(i,3):
        result = quad(lambda x: mij(i,j,x),-1.,1.)
        M[i,j] = result[0]
        M[j,i] = M[i,j]
        R[i,j] = result[1]
        R[j,i] = R[i,j]
        
        
print 'M = {}\n'.format(M)
print 'fhat = {}\n'.format(fhat)        

# and solve for c
c_galerkin = la.solve(M,fhat)

print 'c_galerkin ={}'.format(c_galerkin)



# now plot them all out and compare
plt.figure()
plt.plot(xx,f(xx),'r',label='$f(x)$')
plt.hold(True)
plt.plot(x,f(x),'ro')
plt.plot(xx,v(c_interp,xx),'g',label='$f_{interp}(x)$')
plt.plot(xx,v(c_galerkin,xx),'b',label='$u(x)$')
plt.xlabel('x')
plt.grid()
plt.legend(loc='best')
plt.show()

