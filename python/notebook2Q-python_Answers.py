# libraries required for plotting and maths
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

ns=400

# two circular blobs - requires more hidden layers to fit
cx1=0.3
cy1=0.3
r1=0.3

cx2=-0.4
cy2=-0.4
r2=0.3

# create the data
X=2*(np.random.rand(ns,2)-0.5)
C1=(X[:,0]-cx1)**2 + (X[:,1]-cy1)**2 < r1**2
C2=(X[:,0]-cx2)**2 + (X[:,1]-cy2)**2 < r2**2
T=1.0*(C1 | C2)
T=T[:,None]

# do the plots
plt.figure(figsize=(6,4))
plt.scatter(X[:,0],X[:,1],c=T);

def predict(Xb, w, v):
    assert Xb.shape[1]==w.shape[0]
    def sigmoid(z):
        return 1.0/(1+np.exp(-z))
    ns=Xb.shape[0]
    H=sigmoid(np.matmul(Xb,w))
    Hb=np.hstack([H, np.ones((ns,1))])
    P=sigmoid(np.matmul(Hb,v))
    return P, H

def cost(P, T):
    C = -np.sum(T*np.log(P)+(1-T)*np.log(1-P))/ns
    return C

def gradient(Xb, Hb, v, P, T):
    assert P.shape==T.shape
    ns=Xb.shape[0]
    Δp=P-T
    dCdv=np.matmul(Hb.T, Δp)/ns
    
    M=np.matmul(Δp,v[:-1].T)
    Δh=Hb[:,:-1]*(1-Hb[:,:-1])*M
    dCdw=np.matmul(Xb.T,Δh)/ns
    return dCdw, dCdv

def GradDescent(X, T, nh, α, n_iter):
    
    n_samples=X.shape[0]
    nx=X.shape[1]
    Xb=np.hstack([X,np.ones((n_samples,1))])
    w=np.random.randn(nx+1,nh)
    v=np.random.randn(nh+1,1)
    
    C_i=np.zeros((n_iter,1))
    
    for i in range(n_iter):
        P, H = predict(Xb, w, v)
        Hb=np.hstack([H,np.ones((n_samples,1))])
        C_i[i] = cost(P, T)
        dCdw, dCdv = gradient(Xb, Hb, v, P, T)
        w=w-α*dCdw
        v=v-α*dCdv
        
    return (w, v, C_i)

nh=10
α=2
ni=10000
w,v,Ci = GradDescent(X,T,nh,α,ni)

plt.figure(figsize=(16,9))
plt.subplot(221)
plt.plot(np.arange(Ci.shape[0]), Ci)
plt.xlabel('Iteration')
plt.ylabel('Cost')

plt.subplot(222)
plt.semilogx(np.arange(Ci.shape[0]), Ci)
plt.xlabel('Iteration')
plt.ylabel('Cost');

Xb = np.hstack([X, np.ones((ns,1))])
predictions, hidden = predict(Xb, w, v)
accuracy = np.sum(1*(predictions>0.5)==T)/ns
print("Percetage of correctly classified datapoints:", "{}%".format(100*accuracy))

x = np.linspace(-1,1,100)

y2 = []
for i in range(w.shape[1]):
    y2.append(-x*w[0,i]/w[1,i]-w[2,i]/w[1,i])

plt.figure(figsize=(16,9))
    
plt.subplot(221)
plt.scatter(X[:,0],X[:,1],c=T)
for yy in y2:
    plt.plot(x,yy,'r--')
plt.ylim((-1,1))
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(222)
axis=np.arange(-1,1.1,0.1)
xx, yy =np.meshgrid(axis, axis)
xx = np.matrix.flatten(xx)[:,None]
yy = np.matrix.flatten(yy)[:,None]
XXb = np.hstack([xx,yy,np.ones_like(xx)])
PP, _ = predict(XXb, w, v)
PP = (PP>0.5)
plt.scatter(xx, yy, c=PP)
for yy in y2:
    plt.plot(x,yy,'r--')
plt.ylim((-1,1))
plt.xlabel('x')
plt.ylabel('y');





