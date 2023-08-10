import mosek
from mosek.fusion import *

def buildmodel(A,b):
    M = Model('The model')
        
    (m,n) = A.shape
       
    print('m: %d n: %d' % (m,n))
        
    # Define the variables 
    t     = M.variable('t', 1, Domain.unbounded())
    x     = M.variable('x', n, Domain.unbounded())

    # Define the constraint (t;Ax+b) in quadratic cone
    e     = Expr.vstack(t,Expr.add(Expr.mul(A,x),b))
    M.constraint('con', Expr.transpose(e), Domain.inQCone())

    # Objective
    M.objective('obj',ObjectiveSense.Minimize, t)
       
    return M,t,x 

import numpy

numpy.random.seed(379) # Fix the seed

m       = 10           # Number of rows in A 
n       = 3            # Number of columns in A  
mu      = 100
sigma   = 30

# Generate random test data
A       =  numpy.random.normal(mu, sigma, (m,n))
b       =  numpy.random.normal(mu, sigma, m)

M,t,x   = buildmodel(A,b)

M.solve()

print('Primal objective value: %e Optimal r value: %e' % (M.primalObjValue(),t.level()))
   


def buildregumodel(A,b,lam):
    M = Model('The regularized problem')
        
    (m,n) = A.shape
       
    # Define the variables 
    t     = M.variable('t', 1, Domain.unbounded())
    x     = M.variable('x', n, Domain.unbounded())
    z     = M.variable('z', n, Domain.unbounded())

    # Define the constraint (t;Ax+b) in quadratic cone
    e     = Expr.vstack(t,Expr.add(Expr.mul(A,x),b))
    M.constraint('con', Expr.transpose(e), Domain.inQCone())
    for j in range(n):
        M.constraint('z_%d' % j,Expr.hstack(z.index(j),x.index(j)),Domain.inQCone())
    
    # Objective
    M.objective('obj',ObjectiveSense.Minimize, Expr.add(t,Expr.mul(lam,Expr.sum(z))))
       
    return M,t,x 

lam   = 2.0 # Lambda     
M,t,x = buildregumodel(A,b,lam)

M.solve()

print('Primal objective value: %e Optimal t value: %e' % (M.primalObjValue(),t.level()))

def buildregumodel2(A,b,lam):
    M = Model('The regularized problem')
        
    (m,n) = A.shape
       
    # Define the variables 
    t     = M.variable('t', 1, Domain.unbounded())
    x     = M.variable('x', n, Domain.unbounded())
    z     = M.variable('z', n, Domain.unbounded())

    # Define the constraint (t;Ax+b) in quadratic cone
    e     = Expr.vstack(t,Expr.add(Expr.mul(A,x),b))
    M.constraint('con', Expr.transpose(e), Domain.inQCone())
    M.constraint('z_j >= |x_j|',Expr.hstack(z,x),Domain.inQCone())
    
    # Objective
    M.objective('obj',ObjectiveSense.Minimize, Expr.add(t,Expr.dot(lam,z)))
       
    return M,t,x 

lam   = [2.0]*n # Lambda     
M,t,x = buildregumodel2(A,b,lam)

M.solve()

print('Primal objective value: %e Optimal t value: %e' % (M.primalObjValue(),t.level()))

n   = 2

M   = Model('demo model')

a   = [3.0]*n
 
x   = M.variable('x',n,Domain.unbounded())
y   = M.variable('y',n,Domain.unbounded())
z   = M.variable('z',n,Domain.unbounded())

# Binary version
e0  = Expr.add(x,1.0)           # x+1.0  
e1  = Expr.add(x,y)             # x+y
e2  = Expr.add(a,y)             # a+y
e3  = Expr.sub(x,y)             # x-y 
e4  = Expr.add(Expr.add(x,y),z) # x+y+z

# List version
e5  = Expr.add([x, y, z])       # x+y+z

# Multiplication 
e6  = Expr.mul(7.0,x)           # 7.0*x  
e7  = Expr.mulElm(a,x)          # Diag(a)*x, element wise multiplication

# Inner and outer products
e8  = Expr.dot(a,x)             # a'*x
e9  = Expr.outer(a,x)           # a*x' Outer product 

# Reduction type operations
e10 = Expr.sum(x)

print('e0')
print(e0.toString())
print('e1')
print(e1.toString())
print('e2')
print(e2.toString())
print('e3')
print(e3.toString())
print('e4')
print(e4.toString())
print('e5')
print(e5.toString())
print('e6')
print(e6.toString())
print('e7')
print(e7.toString())
print('e8')
print(e8.toString())
print('e9')
print(e9.toString())
print('e10')
print(e10.toString())

