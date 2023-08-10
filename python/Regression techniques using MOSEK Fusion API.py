from mosek.fusion import *

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rc('savefig',dpi=120)

def plot(title,err,nbins,color):
    plt.xlim([min(err),max(err)])
    plt.title(title)
    plt.grid()
    plt.hist( err,nbins,facecolor=color,histtype='stepfilled')
    plt.show()

def ddot(x,y): return sum([ xx*yy for xx,yy in zip(x,y)])
def error(X,w,y): return [ ddot(X[i], w ) - y[i] for i in range(len(y)) ]

def print_err(name,ee):
    n= len(y)
    absee= [ abs(e) for e in ee]
    print("%s avg= %f ; |min| = %f ; |max| = %f ; |avg| = %f"%(name, sum(ee)/n, min(absee), max(absee), sum(absee)/n))

nbuckets = 20

import random as rnd 

n = 500
m = 2
e= 1

X = [ [rnd.gauss(0.0,e) for j in range(m)] for i in range(n)]
y = [ sum(X[i])+rnd.gauss(0.0,e/2.) for i in range(n)]

def lsq_l1_norm(X,y,n,m):
       
    with Model() as M:
           
        w = M.variable(m, Domain.unbounded())
        t = M.variable(n, Domain.unbounded())
            
    
        M.constraint( Expr.sub(Expr.mul(X,w),t), Domain.lessThan(y))
        M.constraint( Expr.add(t,Expr.mul(X,w)), Domain.greaterThan(y))
            
        M.objective(ObjectiveSense.Minimize, Expr.sum(t))
    
        M.solve()
    
        return w.level()

def lsq_l2_norm(X,y,n,m):
    
    with Model() as M:
           
        w = M.variable(m, Domain.unbounded())
        t = M.variable(1, Domain.unbounded())
    
        M.constraint(Expr.vstack(t, Expr.sub( Expr.mul(X,w), y)), Domain.inQCone(1,n+1))
            
        M.objective(ObjectiveSense.Minimize, t)
       
        M.solve()
    
        return w.level()

w1  = lsq_l1_norm(X,y,n,m)
w2  = lsq_l2_norm(X,y,n,m)

err1 = error(X,w1,y)
err2 = error(X,w2,y)

print_err('l1 : ', err1)
print_err('l2 : ', err2)

plot('1-Norm error',err1,nbuckets,'blue')
plot('2-Norm error',err2,nbuckets,'red')

def lsq_32_reg(X,y,n,m):
    
    with Model() as M:
        w = M.variable('w', m)
        
        t = M.variable('t', m, Domain.unbounded())
        s = M.variable('s', m, Domain.unbounded())
        z = M.variable('z', m, Domain.unbounded())
    

        M.constraint(Expr.vstack( t, Expr.sub(Expr.mul(X,w), y)), Domain.inQCone())        
        M.constraint(Expr.hstack(s, z, t), Domain.inRotatedQCone())
        M.constraint(Expr.hstack(Expr.constTerm(m,1./8.),t, s), Domain.inRotatedQCone())
    
        M.objective(ObjectiveSense.Minimize, Expr.sum(t))

        M.solve()
    
        return w.level()

w3_2  = lsq_32_reg(X,y,n,m)


err3_2 = error(X,w3_2,y)


print_err('3/2 linear regression: ', err3_2)

plot('3/2 linear regression',err3_2,nbuckets,'cyan')

def lsq_deadzone_linear(X,y,n,m,a):
    
    if a<0.: return None
    
    with Model() as M:
           
        w = M.variable(m)
        t = M.variable(n)
        
        y_plus_a  = [y[i]+a for i in range(n)]
        y_minus_a = [y[i]-a for i in range(n)]
    
       
        M.constraint( Expr.sub( Expr.mul(X,w), t), Domain.lessThan(y_plus_a))
        M.constraint( Expr.add( t, Expr.mul(X,w)), Domain.greaterThan(y_minus_a))
        
        M.objective(ObjectiveSense.Minimize, Expr.sum(t))
    
        M.solve()
    
        return w.level()

wdzl  = lsq_deadzone_linear(X,y,n,m,1.)


errdzl = error(X,wdzl,y)


print_err('Dead-zone linear regression: ', errdzl)

plot('Dead-zone linear regression',errdzl,10,'green')

def lsq_minmax_norm(X,y,n,m):
    
    with Model() as M:
           
        w = M.variable(m)
        t = M.variable(1, Domain.unbounded())
    
        one_t= Var.repeat(t,n)
    
        M.constraint( Expr.sub( Expr.mul(X,w), one_t), Domain.lessThan(y))
        M.constraint( Expr.add( one_t, Expr.mul(X,w)), Domain.greaterThan(y))
        
        M.objective(ObjectiveSense.Minimize, t)
    
        M.solve()
    
        return w.level()

wmm = lsq_minmax_norm(X,y,n,m)


errmm = error(X,wmm,y)


print_err('Minimax regression: ', errmm)

plot('Minimax regression',errmm,10,'orange')


def lsq_k_largest(X,y,n,m,k):
    
    with Model() as M:
        
        w = M.variable(m, Domain.unbounded())
        r = M.variable(n, Domain.unbounded())
        t = M.variable(n, Domain.greaterThan(0.))
        z = M.variable(1, Domain.unbounded())
        s = M.variable(1, Domain.unbounded())
        u = M.variable(n, Domain.greaterThan(0.))
                  
        M.constraint(Expr.sub(Expr.mul(X,w),r), Domain.equalsTo(y))
        
        M.constraint( Expr.sub(t,r), Domain.greaterThan(0.))
        M.constraint( Expr.add(t,r), Domain.greaterThan(0.))
        
        M.constraint( Expr.add([Expr.mul(float(k),s),Expr.sum(u),Expr.neg(z)]), Domain.lessThan(0.))
        
        M.constraint( Expr.sub(Expr.add(Var.repeat(s,n),u), t), Domain.greaterThan(0.))
        
        M.objective(ObjectiveSense.Minimize, z)
        M.solve()
    
        return w.level()

wkl = lsq_k_largest(X,y,n,m,10.)


errkl = error(X,wkl,y)


print_err('K-largest regression: ', errkl)

plot('K-largest regression',errkl,10,'yellow')


get_ipython().magic('timeit w1  = lsq_l1_norm(X,y,n,m)')
get_ipython().magic('timeit w2  = lsq_l2_norm(X,y,n,m)')
get_ipython().magic('timeit w32 = lsq_32_reg(X,y,n,m)')
get_ipython().magic('timeit wmm = lsq_minmax_norm(X,y,n,m)')
get_ipython().magic('timeit wkl = lsq_k_largest(X,y,n,m, 10)')
get_ipython().magic('timeit wdzl= lsq_deadzone_linear(X,y,n,m, 0.15)')

