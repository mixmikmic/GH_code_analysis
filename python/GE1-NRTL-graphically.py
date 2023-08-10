import numpy as np

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

#unit system constant
from scipy.constants import R
print('R = ',R)

#trial temperature and composition:
T = 293.15 #K
x=np.array([.1,.3,.6]) #normalized

# Ethyl acetate (1) + water (2) + ethanol (3)

alpha12 = 0.4

alpha23 = 0.3

alpha13 = 0.3

# 6 binary Aij parameters
Dg12 = 1335 * 4.184 #J/K
Dg21 = 2510 * 4.184 #J/K

Dg23 = 976 * 4.184 #J/K
Dg32 = 88 * 4.184 #J/K

Dg13 = 301 * 4.184 #J/K
Dg31 = 322 * 4.184 #J/K

#assemble matrix with regressed parameters Dg_i,j, according to the model all diagonal terms are zero
Dg = np.array([[0, Dg12, Dg13],
             [Dg21, 0, Dg23],
             [Dg31, Dg32, 0]])


#assemble symmetric matrix alpha
alpha = np.array([[0, alpha12, alpha13],
                [alpha12, 0, alpha23],
                [alpha13, alpha23, 0]])

#verify the assembled matrices

#we can now automatically loop through its elements
print('loop through its elements')
print("i,j,alpha[i,j],Dg[i,j]")
for i in range(3):
    for j in range(3):
        print(i,j,alpha[i,j],Dg[i,j])
        
#or even use the whole matrix at once
print('whole matrix')
print("alpha=")
print(alpha)
print("Dg=")
print(Dg)

# rather than typing each element individually
print('typing each element')
print("alpha12,alpha13,alpha23")
print(alpha12,alpha13,alpha23)
print("Dg12,Dg13,Dg21,Dg23,Dg31,Dg32")
print(Dg12,Dg13,Dg21,Dg23,Dg31,Dg32)

#finally calculate parameter A in units of K-1 from Dg and the constant R
A = Dg/R 

tau=np.zeros([3,3])
for j in range(3):
    for i in range(3):
        tau[j,i]=A[j,i]/T
print("tau=")
print(tau)
        
G=np.zeros([3,3])
for j in range(3):
    for i in range(3):
        G[j,i]=np.exp((-alpha[j,i]*tau[j,i]))
print("G=")
print(G)

Gamma=np.zeros([3])
for i in range(3):

    Sj1=0
    Sj2=0
    Sj3=0
    for j in range(3):
        Sj1     += tau[j,i]*G[j,i]*x[j]
        Sj2     += G[j,i]*x[j]

        Sk1=0
        Sk2=0
        Sk3=0
        for k in range(3):
            Sk1+=G[k,j]*x[k]
            Sk2+=x[k]*tau[k,j]*G[k,j]
            Sk3+=G[k,j]*x[k]
        
        Sj3     += ((x[j]*G[i,j])/(Sk1))*(tau[i,j]-(Sk2)/(Sk3))
    
    Gamma[i]=np.exp(Sj1/Sj2 + Sj3)
    
print(Gamma)

def Gamma(T,x,alpha,A):
    tau=np.zeros([3,3])
    for j in range(3):
        for i in range(3):
            tau[j,i]=A[j,i]/T    
    
    G=np.zeros([3,3])
    for j in range(3):
        for i in range(3):
            G[j,i]=np.exp((-alpha[j,i]*tau[j,i]))
    
    Gamma=np.zeros([3])
    for i in range(3):

        Sj1=0
        Sj2=0
        Sj3=0
        for j in range(3):
            Sj1 += tau[j,i]*G[j,i]*x[j]
            Sj2 += G[j,i]*x[j]
    
            Sk1=0
            Sk2=0
            Sk3=0
            for k in range(3):
                Sk1 += G[k,j]*x[k]
                Sk2 += x[k]*tau[k,j]*G[k,j]
                Sk3 += G[k,j]*x[k]
            
            Sj3 += ((x[j]*G[i,j])/(Sk1))*(tau[i,j]-(Sk2)/(Sk3))
        
        Gamma[i]=np.exp(Sj1/Sj2 + Sj3)
    
    return Gamma

#test it to see if results match
ans=Gamma(T,x,alpha,A)
print(ans) #ttest using those trial input

# test predictions of activity coefficients at infinite dillution
ans=Gamma(T,[1,0,0],alpha,A)
print("in ~pure Ethyl acetate liquid, Ethyl acetate's itself activity coefficient is ",ans[0])
print("ethanol infinite dillution activity coefficient is ", ans[2])
print("and water infinite dillution activity coefficient is ", ans[1])

ans=Gamma(T,[0,1,0],alpha,A)

print("in ~pure ethanol liquid, ethanol's itself activity coefficient is ",ans[2])
print("Ethyl acetate infinite dillution activity coefficient is ", ans[0])
print("and water infinite dillution activity coefficient is ", ans[1])

ans=Gamma(T,[0,0,1],alpha,A)

print("in ~pure water liquid, water's itself activity coefficient is ",ans[1])
print("ethanol infinite dillution activity coefficient is ", ans[2])
print("and Ethyl acetate infinite dillution activity coefficient is ", ans[0])

# Ethyl acetate (1) + water (2) + ethanol (3)

#consider a mixture of (2) and (3)

T=298

xEA=np.zeros(100)
xW=np.linspace(0,1,100)
xE=1-xEA-xW

GE = np.zeros(100)
GM = np.zeros(100)
GIM = np.zeros(100)


GIM[0]=0
GIM[99]=0
for i in range(1,99):  # from 1 to 98, inclusive
    GIM[i]=(R*T*(xW[i]*np.log(xW[i])+
                 xE[i]*np.log(xE[i])))


for i in range(100):
    
    gammas=Gamma(T,[xEA[i],xW[i],xE[i]],alpha,A)
    
    GE[i]=(R*T*(xW[i]*np.log(gammas[1])+
                xE[i]*np.log(gammas[2])))
    
    GM[i]=GIM[i]+GE[i]

plt.ylabel(r'Gibbs Energy variations')
plt.xlabel(r'$x_W$')
plt.plot(xW,GE,label=r'$G^E$')
plt.plot(xW,GM,label=r'$G^{M}$')
plt.plot(xW,GIM,label=r'$G^{IM}$')
plt.legend(loc=4)
plt.show()

#for a 3 component system we can conceive 3 binary

import itertools as it
binaries = list(it.combinations([0, 1, 2],2))

print(binaries)

# figs = np.ndarray(3,dtype=object)
# axs = np.ndarray(4,dtype=object)

GE = np.asarray([np.zeros(100),np.zeros(100),np.zeros(100)])
GM = np.asarray([np.zeros(100),np.zeros(100),np.zeros(100)])
GIM = np.asarray([np.zeros(100),np.zeros(100),np.zeros(100)])

for ibin in range(3):

#    figs[ibin], axs[ibin] = plt.subplots(1,1)

    
    T=298

    x=np.zeros([100,3])
    
#    print(binaries[ibin][0])
    compA=binaries[ibin][0]
#    print(binaries[ibin][1])
    compB=binaries[ibin][1]
    
    x[:,compA]=np.linspace(0,1,100)
    x[:,compB]=1-x[:,compA]
    
#    print(x[0,:])    
#    print(x[1,:])    
#    print(x[50,:])    
#    print(x[98,:])    
#    print(x[99,:])    

#    GIM[0]=0
#    GIM[99]=0

    for i in range(1,99):  # from 1 to 98, inclusive
        GIM[ibin][i]=(R*T*(x[i,compA]*np.log(x[i,compA])+
                     x[i,compB]*np.log(x[i,compB])))

    
    for i in range(100):
        gammas=Gamma(T,x[i,:],alpha,A)
        GE[ibin][i]=(R*T*(x[i,compA]*np.log(gammas[compA])+
                    x[i,compB]*np.log(gammas[compB])))
        GM[ibin][i]=GIM[ibin][i]+GE[ibin][i]
        
#    axs[ibin].set_ylabel(r'Gibbs Energy variations')
#    axs[ibin].set_xlabel(r'$x_W$')
#    axs[ibin].set_title(str(compA)+str(compB))
#    axs[ibin].plot(x[:,compA],GE[ibin],label=r'$G^E$')
#    axs[ibin].plot(x[:,compA],GM[ibin],label=r'$G^{M}$')
#    axs[ibin].plot(x[:,compA],GIM[ibin],label=r'$G^{IM}$')
#    axs[ibin].legend(loc=4)

    
# plt.show()
        

#for a 3 component system we can conceive 3 binary

#import itertools as it
#binaries = list(it.combinations([0, 1, 2],2))

#print(binaries)

figs = np.ndarray(3,dtype=object)
axs = np.ndarray(4,dtype=object)

#GE = np.asarray([np.zeros(100),np.zeros(100),np.zeros(100)])
#GM = np.asarray([np.zeros(100),np.zeros(100),np.zeros(100)])
#GIM = np.asarray([np.zeros(100),np.zeros(100),np.zeros(100)])

for ibin in range(3):

    figs[ibin], axs[ibin] = plt.subplots(1,1)

    
#    T=298

#    x=np.zeros([100,3])
    
#    print(binaries[ibin][0])
#    compA=binaries[ibin][0]
#    print(binaries[ibin][1])
#    compB=binaries[ibin][1]
    
#    x[:,compA]=np.linspace(0,1,100)
#    x[:,compB]=1-x[:,compA]
    
#    print(x[0,:])    
#    print(x[1,:])    
#    print(x[50,:])    
#    print(x[98,:])    
#    print(x[99,:])    

#    GIM[0]=0
#    GIM[99]=0

#    for i in range(1,99):  # from 1 to 98, inclusive
#        GIM[ibin][i]=(R*T*(x[i,compA]*np.log(x[i,compA])+
#                     x[i,compB]*np.log(x[i,compB])))

    
#    for i in range(100):
#        gammas=Gamma(T,x[i,:],alpha,A)
#        GE[ibin][i]=(R*T*(x[i,compA]*np.log(gammas[compA])+
#                    x[i,compB]*np.log(gammas[compB])))
#        GM[ibin][i]=GIM[ibin][i]+GE[ibin][i]
        
    axs[ibin].set_ylabel(r'Gibbs Energy variations')
    axs[ibin].set_xlabel(r'$x_W$')
    axs[ibin].set_title(str(compA)+str(compB))
    axs[ibin].plot(x[:,compA],GE[ibin],label=r'$G^E$')
    axs[ibin].plot(x[:,compA],GM[ibin],label=r'$G^{M}$')
    axs[ibin].plot(x[:,compA],GIM[ibin],label=r'$G^{IM}$')
    axs[ibin].legend(loc=4)

    
plt.show()
        

#we can make 1 meshgrid

xEA = np.linspace(0,1,10)
xW = np.linspace(0,1,10)
xE = np.linspace(0,1,10)

rawxizes = np.ndarray((10,10,10), dtype=object)

flags = np.zeros((10,10,10))

for i in range(10):
    for j in range(10):
        for k in range(10):
            rawxizes[i,j,k]=np.array([xEA[i],xW[j],xE[k]])
            if ( rawxizes[i,j,k][0] + rawxizes[i,j,k][1] +  rawxizes[i,j,k][2] == 1 ):
                flags[i,j,k] = 1
            
xizes = rawxizes[np.where(flags==1)]

npts = xizes.shape[0]
print(npts)

print(xizes[0])
print(xizes[0][2])

GE = np.zeros(npts)

for l in range(npts):
    gammas=Gamma(T,np.array(xizes[l]),alpha,A)
    GE[l] = R*T*(xizes[l][0]*np.log(gammas[0])+
                 xizes[l][1]*np.log(gammas[1])+
                 xizes[l][2]*np.log(gammas[2]))

    def xlogx(x):
        if x==0:
            return 0
        else:
            return x*np.log(x)
    
    GIM[l] = R*T*(xlogx(xizes[l][0])+
                  xlogx(xizes[l][1])+
                  xlogx(xizes[l][2]))

    GM[l] = GE[l]+GIM[l]
            
print(rawxizes[0,0,0])
print(rawxizes[3,5,7])
print(rawxizes[9,9,9])

print(xizes[0])
print(xizes[1])
print(xizes[10])
print(xizes[42])
print(xizes[43])
            

import matplotlib.tri as tri

# first load some data:  format x1,x2,x3,value
print('testdata')
print(xizes[0])
print(GM[0])

test_data = np.zeros([npts,4])

for l in range(npts):
    test_data[l,0:3] = xizes[l][:]
    test_data[l,3] = GM[l]

print(test_data[0])


# barycentric coords: (a,b,c)
a=test_data[:,0]
b=test_data[:,1]
c=test_data[:,2]

# values is stored in the last column
v = test_data[:,-1]

# create a triangulation out of these points
#Tri = tri.Triangulation(cartx,carty)
Tri = tri.Triangulation(a,b)

# plot the contour
#plt.tricontourf(cartx,carty,Tri.triangles,v)
plt.tricontourf(a,b,Tri.triangles,v)


# create the grid
#corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
corners = np.array([[0, 0], [1, 0], [0,1]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

# creating the grid
refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

#plotting the mesh
plt.triplot(trimesh,'k--')

plt.title('GM')
#plt.axis('off')
plt.show()

#for a 3 component system we can conceive 3 binary

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import itertools as it
binaries = list(it.combinations([0, 1, 2],2))

print(binaries)

figs = np.ndarray(3,dtype=object)
axs = np.ndarray(4,dtype=object)

fig, ((axs[0],axs[1]),(axs[2],axs[3])) = plt.subplots(2,2)

for ibin in range(3):
    
#    figs[ibin], axs[ibin] = plt.subplots(1,1)
    
    T=298

    x=np.zeros([100,3])
    
#    print(binaries[ibin][0])
    compA=binaries[ibin][0]
#    print(binaries[ibin][1])
    compB=binaries[ibin][1]
    
    x[:,compA]=np.linspace(0,1,100)
    x[:,compB]=1-x[:,compA]
    
#    print(x[0,:])    
#    print(x[1,:])    
#    print(x[50,:])    
#    print(x[98,:])    
#    print(x[99,:])    
    
    GE = np.zeros(100)
    GM = np.zeros(100)
    GIM = np.zeros(100)

    GIM[0]=0
    GIM[99]=0
    for i in range(1,99):  # from 1 to 98, inclusive
        GIM[i]=(R*T*(x[i,compA]*np.log(x[i,compA])+
                     x[i,compB]*np.log(x[i,compB])))

    
    for i in range(100):
        gammas=Gamma(T,x[i,:],alpha,A)
        GE[i]=(R*T*(x[i,compA]*np.log(gammas[compA])+
                    x[i,compB]*np.log(gammas[compB])))
        GM[i]=GIM[i]+GE[i]
        
    axs[ibin].set_ylabel(r'Gibbs Energy variations')

    axs[ibin].set_title(str(compA)+str(compB))
    axs[ibin].plot(x[:,compA],GE,label=r'$G^E$')
    axs[ibin].plot(x[:,compA],GM,label=r'$G^{M}$')
    axs[ibin].plot(x[:,compA],GIM,label=r'$G^{IM}$')
    axs[ibin].legend(loc=4)

axs[0].set_xlabel(r'$x_{EA}$')
axs[1].set_xlabel(r'$x_W$')
axs[2].set_xlabel(r'$x_E$')

axs[3].set_xlabel(r'$x_{EA}$')
axs[3].set_ylabel(r'$x_{W}$')

# barycentric coords: (a,b,c)
a=test_data[:,0]
b=test_data[:,1]
c=test_data[:,2]

# values is stored in the last column
v = test_data[:,-1]

# translate the data to cartesian corrds
#cartx = 0.5 * ( 2.*b+c ) / ( a+b+c )
#carty = 0.5*np.sqrt(3) * c / (a+b+c)


# create a triangulation out of these points
#Tri = tri.Triangulation(cartx,carty)
Tri = tri.Triangulation(a,b)

# plot the contour
#axs[3].tricontourf(cartx,carty,Tri.triangles,v)
axs[3].tricontourf(a,b,Tri.triangles,v)


# create the grid
#corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
corners = np.array([[0, 0], [1, 0], [0,1]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

# creating the grid
refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

#plotting the mesh
axs[3].triplot(trimesh,'k--')

axs[3].set_title('GM')
#axs[3].axis('off')

plt.tight_layout()

axs[3].spines['right'].set_visible(False)
axs[3].spines['top'].set_visible(False)
axs[3].yaxis.set_ticks_position('left')
axs[3].xaxis.set_ticks_position('bottom')

plt.show()
        







