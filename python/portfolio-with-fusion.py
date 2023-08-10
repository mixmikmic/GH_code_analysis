n  = 3

mu = [0.1073, 0.0737, 0.0627]

w  = 1.0

# stored in a single 1-D array in column-major order
Sigma = [    0.02778 ,   0.00387 ,  0.00021,    0.00387 ,   0.01112 , -0.00020,    0.00021 ,  -0.00020 ,  0.00115     ]

import mosek as msk

with msk.Env() as env:
    env.potrf( msk.uplo.lo,n, Sigma)
    
#copying in a 2D array and zeroing out the lower triangular part
GT = [ [ 0. if j<i else Sigma[i+j*n] for j in range(n)] for i in range(n)]
    
for i in range(n):
    print "| %s |" % ','.join(['%+.4f'% GT[i][j] for j in range(n)])

from mosek.fusion import *

def EfficientFrontier(mu,GT,w,alphas):
    
    n = len(mu)
    
    with Model("Efficient frontier") as M:

        # Defines the variables (holdings). Shortselling is not allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0)) # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded()) # Risk variable

        M.constraint('budget', Expr.sum(x), Domain.equalsTo(w))

        M.constraint('risk', Expr.vstack(s,Expr.mul(GT,x)),Domain.inQCone())

        risk = []
        xs = []
        
        mudotx = Expr.dot(mu,x)
        for alpha in alphas:

            #  Define objective as a weighted combination of return and risk
            M.objective('obj', ObjectiveSense.Maximize, Expr.sub(mudotx,Expr.mul(alpha,s)))

            M.solve()

            risk.append(s.level()[0])
            xs.append(x.level())
            
        return risk,xs

alphas = [0.00, 0.01, 0.10, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00]

risk,xs = EfficientFrontier(mu,GT,w,alphas)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.rc('savefig',dpi=120)

returns = [ sum([ m*x for m,x in zip(mu,xx)]) for xx in xs]
plt.plot( risk, returns , 'b*')
plt.grid()    
plt.xlabel('Risk')
plt.ylabel('Return')
plt.title("Return vs. Risk: the efficient frontier")

plt.plot( alphas,  [ sum([ m*x for m,x in zip(mu,xx)]) for xx in xs], 'b*')
plt.grid()    
plt.xlabel('Alpha')
plt.ylabel('Return')

m = len(alphas)
indx= range(m)
colors=['b','r','g']
bott = [0. for i in indx]

for i in range(n):
    xx = [ xs[j][i] for j in indx]
    plt.bar(indx, xx ,  0.35, bottom=bott,color=colors[i])
    bott= [ bott[j] + xx[j] for j in indx]

plt.xticks(indx, alphas)
plt.xlabel('alpha')
plt.ylabel('portfolio')
plt.grid()
plt.show()



