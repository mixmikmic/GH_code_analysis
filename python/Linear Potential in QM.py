import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
hbar=1
m=1
N = 4096
a = 15.0

# This is for the symmetric linear potential
xs = np.linspace(-a/2.,a/2.,N)
Vs = np.abs(xs)
# This is for the one-sided linear potential
xo = np.linspace(0.,a/2.,N/2)
Vo = np.abs(xo)
# Make Plots
fig1 = plt.figure(figsize=(8,6))
# plt.xkcd()    # Set hand drawn looking style
#plt.xticks([]) # And remove x and y ticks.
#plt.yticks([]) # For plotting.
plt.plot([0,0],[-2,a/2.],color="blue")  # Draw the axes in blue.
plt.plot([-a/2.,a/2.],[0,0],color="blue")
plt.plot(xs,Vs,color="green")           # Plot the potential in green
plt.title("Symmetric Linear Potential")
plt.savefig("Symmetric_Linear_potential.pdf")
plt.show()
#
# Now plot the one-sided case
#
fig1 = plt.figure(figsize=(8,6))
#plt.xticks([])
#plt.yticks([])
plt.plot([0,0],[-2,a/2.],color="blue")
plt.plot([0,a/2.],[0,0],color="blue")
plt.plot([0,0],[0,a/2.],color="green") # Plot the infinity side. 
plt.plot(xo,Vo,color="green")
plt.title("One Sided Linear Potential")
plt.savefig("Onesided_Linear_potential.pdf")
plt.show()

# This is for the Symmetric linear potential case.
hs = xs[1]-xs[0] # Should be equal to 2*np.pi/(N-1)
Mdds = 1./(hs*hs)*(np.diag(np.ones(N-1),-1) -2* np.diag(np.ones(N),0) + np.diag(np.ones(N-1),1))
Hs = -(hbar*hbar)/(2.0*m)*Mdds + np.diag(Vs) 
Es,psiTs = np.linalg.eigh(Hs) # This computes the eigen values and eigenvectors
psis = np.transpose(psiTs) 
# We now have the eigen vectors as psi(i), where i is the energy level.
print np.sum(psis[0]*psis[0])  # Check. Yes these are normalized already.

# This is for the One sided case.

ho = xo[1]-xo[0] # Should be equal to 2*np.pi/(N-1)
Mddo = 1./(ho*ho)*(np.diag(np.ones(N/2-1),-1) -2* np.diag(np.ones(N/2),0) + np.diag(np.ones(N/2-1),1))
Ho = -(hbar*hbar)/(2.0*m)*Mddo + np.diag(Vo) 
Eo,psiTo = np.linalg.eigh(Ho) # This computes the eigen values and eigenvectors
psio = np.transpose(psiTo)
# We now have the eigen vectors as psi(i), where i is the energy level.
print np.sum(psio[0]*psio[0])  # Check. Yes these are normalized already.
# print psiT[0] # Uncomment to see the values printed for Psi_0

print "Symmetric Case  \t One-sided Case"
for n in range(12):
    if n%2==1:
        no = (n-1)/2
        print "Es[{}] = {:9.4f}\t Eo[{}] ={:9.4f}".format(n,Es[n],no, Eo[no])
    else:
        print "Es[{}] = {:9.4f} ".format(n,Es[n])


plt.figure(figsize=(10,6))
plt.plot(xs,0.1*Vs,color="grey",label="Potential: 0.1V(x)")
plt.ylim((-0.9,0.9))
for i in range(6):
    if psis[i,N-10]<0:
        plt.plot(xs,-np.real(psis[i])/np.sqrt(hs),label="E_{}={:8.4f}".format(i,Es[i]))
    else:
        plt.plot(xs,np.real(psis[i])/np.sqrt(hs),label="E_{}={:8.4f}".format(i,Es[i]))
    plt.legend()
    plt.title("Solutions to the Linear Potential")
    plt.xlabel("x")
    plt.ylabel("$\psi(x)$")
plt.savefig("Linear_Potential_Wavefunctions.pdf")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(xo,0.1*Vo,color="grey",label="Potential: 0.1V(x)")
plt.ylim((-0.9,0.9))
for i in range(6):
    if psio[i,N/2-10]<0:
        plt.plot(xo,-psio[i]/np.sqrt(ho),label="E_{}={}".format(i,Eo[i]))
    else:
        plt.plot(xo,psio[i]/np.sqrt(ho),label="E_{}={}".format(i,Eo[i]))
    plt.legend()
    plt.title("Solutions to the One Sided Linear Potential")
    plt.xlabel("x")
    plt.ylabel("$\psi(x)$")
plt.savefig("One_Sided_Linear_Potential_Wavefunctions.pdf")
plt.show()

