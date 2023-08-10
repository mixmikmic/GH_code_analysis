import numpy as np
import numpy.linalg as la

v = np.linspace(0,1,11)
n = len(v)
A = np.concatenate((2*v.reshape((1,n)), np.ones((1,n))), axis=0)
c = 1+v**2
b = np.array([1,1])

# Define the function F and the Jacobian matrix M
def F(x, y, s):
    C1 = np.dot(A.T,y)+s-c
    C2 = np.dot(A,x)-b
    C3 = x*s
    return np.concatenate((C1, C2, C3))

def M(x, y, s):
    return np.asarray(np.bmat([[np.zeros((n,n)), A.T, np.eye(n)],
                    [A, np.zeros((2,2)), np.zeros((2,n))],
                    [np.diag(s), np.zeros((n,2)), np.diag(x)]]))

x = np.ones(n)/11.
y = np.array([0,0])
s = c-np.dot(A.T, y)

def longstep(x, y, s, sigma, gamma=1e-3, tol=1e-4): 
    mu = 1
    i = 1
    yy = np.zeros((2,50))
    while mu>tol and i<50:
        a = 1
        mu = np.dot(x,s)/11.
        rhs = F(x,y,s)-np.concatenate((np.zeros(n+2), sigma*mu*np.ones(11)))
        delta = -la.solve(M(x,y,s), rhs)
        xs = np.concatenate((x,s))
        deltaxs = np.concatenate((delta[:11], delta[13:]))
    
        I = np.argmin(xs+deltaxs)
        m = xs[I]+deltaxs[I]
        if m<gamma*mu:
            a = np.amin(-xs[I]/deltaxs[I])
    
        x = x+a*delta[:11]
        y = y+a*delta[11:13]
        s = s+a*delta[13:]
    
        yy[:,i] = y
        i+=1
    return yy[:,:i]

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(1,3, figsize=(12, 4))
xx = np.linspace(0,1,100)
sigmas = [0.1, 0.5, 0.9]

for k in range(3):
    yy = longstep(x,y,s,sigmas[k])
    ax[k].set_ylim([0,1])
    for j in range(n):
        ax[k].plot(xx,c[j]-np.dot(A[0,j],xx))
    ax[k].plot(yy[0,:], yy[1,:], '-o')
    ax[k].set_title("$\sigma$={:f}".format(sigmas[k]))
    ax[k].set_xlabel('$y_1$')
    ax[k].set_ylabel('$y_2$')
plt.show()

def longstep_xs(x, y, s, sigma, gamma=1e-3, tol=1e-4): 
    mu = 1
    i = 1
    xxs = np.zeros((2,50))
    xxs[:,0] = np.array([x[1]*s[1],x[4]*s[4]])
    while mu>tol and i<50:
        a = 1
        mu = np.dot(x,s)/11.
        rhs = F(x,y,s)-np.concatenate((np.zeros(n+2), sigma*mu*np.ones(11)))
        delta = -la.solve(M(x,y,s), rhs)
        xs = np.concatenate((x,s))
        deltaxs = np.concatenate((delta[:11], delta[13:]))
    
        I = np.argmin(xs+deltaxs)
        m = xs[I]+deltaxs[I]
        if m<gamma*mu:
            a = np.amin(-xs[I]/deltaxs[I])
    
        x = x+a*delta[:11]
        y = y+a*delta[11:13]
        s = s+a*delta[13:]
        xxs[:,i] = np.array([x[1]*s[1],x[4]*s[4]])
        i+=1
    return xxs[:,:i]

fig, ax = plt.subplots(1,3, figsize=(12, 4))
xx = np.linspace(0,0.1,100)
sigmas = [0.1, 0.5, 0.9]

for k in range(3):
    xs = longstep_xs(x,y,s,sigmas[k])
    ax[k].plot(xx,xx,linewidth=3, color='red')
    ax[k].plot(xs[0,:], xs[1,:], '-o')
    ax[k].set_title("$\sigma$={:f}".format(sigmas[k]))
    ax[k].set_xlabel('$x_1s_1$')
    ax[k].set_ylabel('$x_5s_5$')
plt.show()

S = np.array([[185, 86.5, 80, 20],
    [86.5, 196, 76, 13.5],
    [80, 76, 411, -19],
    [20, 13.5, -19, 25]])
r = np.array([14,12,15,7])
e = np.ones(4)

Se = la.solve(S, e)
Sr = la.solve(S, r)
a = np.dot(e, Se)
b = np.dot(e, Sr)
c = np.dot(r, Sr)
d = (c*Sr-b*Se)/(a*c-b**2)
s = (a*Se-b*Sr)/(a*c-b**2)

mu = np.linspace(0,30,300)
xx = np.outer(d.reshape((len(d),1)),np.ones((len(mu),1)))+np.outer(s.reshape((len(s),1)),mu.reshape((len(mu),1)))
temp = np.dot(S,xx)
risk = np.zeros(xx.shape[1])
for i in range(xx.shape[1]):
    risk[i] = np.dot(xx[:,i],temp[:,i])
plt.plot(mu, risk, linewidth=3)
plt.show()

