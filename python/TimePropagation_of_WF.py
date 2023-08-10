import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
import scipy.fftpack as fft
# Input parameters.
hbar=1
m=1
N = 2**12   # FFT is a lot faster on domains which are powers of 2.
a = 400.0
step_low = 0.
step_high= 1.
V0 = 10.
# Define the space:
n = np.arange(N)
x0 = -a/2.
xN = a/2.
Delta_x = (xN - x0)/N
print "Delta_x = ",Delta_x
x = x0 + n*Delta_x  # This creates the correct array of x values.
###################################################
# Potential Definition.
###################################################
V=np.zeros(N)
for i in range(N):
    if x[i]>= step_low and x[i]<= step_high:
        V[i]= V0
###################################################
# Setup the Hamiltonian for function V
#
Mdd = 1./(Delta_x**2)*(np.diag(np.ones(N-1),-1) -2* np.diag(np.ones(N),0) + np.diag(np.ones(N-1),1))
H = -(hbar*hbar)/(2.0*m)*Mdd + np.diag(V) 
En,psiT = np.linalg.eigh(H) # This computes the eigen values and eigenvectors
psi = np.transpose(psiT)   # We take the transpose of psiT to the wavefunction vectors can accessed as psi[n]

dt_max = 2/np.max(En) # Stability Criteria
dt= 0.001
if dt>dt_max:
    print "WARNING: dt is in the instable region!!!!!"

g_x0=-10.
g_k0=6.
g_sig=2.

# Gaussian in k-space, where p = \hbar k, with a momentum k0, and x-space psi(x,0) (2a/pi)^1/4 e^{-ax^2}
def psi0(x,g_x0,g_k0,g_sig):
    _Norm_x=np.sqrt(Delta_x/g_sig)/(np.pi**0.25)
    return(_Norm_x*np.exp(-(x-g_x0)**2/(2.*g_sig*g_sig)+1j*g_k0*x))

psi_t0 = psi0(x,g_x0,g_k0,g_sig)

#def phi(k,g_k0,g_x0,g_sig):
#    _Norm_k=np.sqrt(g_sig*Delta_k)/(np.pi**0.25)
#    return(_Norm_k*np.exp(-0.5*g_sig*g_sig*(k-g_k0)**2-1j*g_x0*(k-g_k0)))

print "Check if H is actually Hermetian! ",np.array_equal(H.conj().T,H) # Is H Hermetian?

Ut_mat = np.diag(np.ones(N,dtype="complex128"),0)
print "Creating a U(dt={}) Matrix".format(dt)
for n in range(1,3):  # Do the summation. Since this is matrixes, this takes a little while if N is large.
    Ut_mat += np.linalg.matrix_power((-1j*dt*H/hbar),n)/math.factorial(n)

p=Ut_mat.dot(psi_t0)
print "How much is the normalization changing per step? From {} to {}".format(np.linalg.norm(psi_t0),np.linalg.norm(p))
print "Number of steps at which point norm is off by a factor of 2: ",1/(np.linalg.norm(p)-1)

Ut_mat_dagger = Ut_mat.conj().T
print "If the U(t) matrix properly unitary? ",np.array_equal(Ut_mat_dagger,Ut_mat)
print "Didn't think so, or else the norm would not change!"
print "This should be the Identity matrix, but the diagonal terms are not 1!!"
print np.diag(Ut_mat.dot(Ut_mat_dagger))[0:5]

psi_t0 = psi0(x,g_x0,g_k0,g_sig)
psi_t1 = psi_t0
psi_tu = []
for t in range(3500):
    psi_t1 = Ut_mat.dot(psi_t1)
    if t>0 and t%500==0:
        psi_tu.append( (t,psi_t1))
psi_tu.append( (t,psi_t1))

print "Normalization: ",np.linalg.norm(psi_tu[-1][1])
vev_E0=float(np.real(np.sum(np.conjugate(psi_t0)*H.dot(psi_t0))))
vev_x0=float(np.real(np.sum(np.conjugate(psi_t0)*x*psi_t0)))
print "<E_(t=0)> = {:8.4f}  <x_(t=0)>= {:8.4f}".format(vev_E0,vev_x0)

for t,p in psi_tu:
    norm = np.linalg.norm(p)
    vev_E1=float(np.real(np.sum(np.conjugate(p)*H.dot(p))))
    vev_x1=float(np.real(np.sum(np.conjugate(p)*x*p)))
    print "dt={:7.1f}  norm = {:8.5f}  <E> = {:8.4f}  <x_(dt)>= {:8.4g}".format(t,norm,vev_E1,vev_x1)

fig,ax1 = plt.subplots(figsize=[10,7])
if vev_E0>max(V):
    plt.title('Scattering')
else:
    plt.title('Tunneling')
ax1.set_ylabel('$\psi(x,t)$')
ax1.set_xlabel('$x$')
#ax1.plot(x,np.abs(psi_t0)/np.sqrt(Delta_x),label="$\Psi(x,t=0)$")
for t,p in psi_tu:
    ax1.plot(x,np.abs(p)/np.sqrt(Delta_x),label="$\Psi(x,t={:6.3f})$".format(t*dt))
ax1.legend(loc='upper left')
ax2=ax1.twinx()
ax2.plot(x,V,color="grey",label="V(x)")
ax2.plot([x[0],x[N-1]],[vev_E0,vev_E0],color="grey",linestyle="--",label="<E>")
ax2.set_ylabel("Energy",color="grey")
ax2.set_xlim(g_x0-5*g_sig,-g_x0+5*g_sig)
ax2.legend(loc='upper right')
#plt.savefig("Displaced_state.pdf")
plt.show()

get_ipython().run_line_magic('time', 'Ut_05s = np.linalg.matrix_power(Ut_mat,int(0.5/dt) )')

psi_t0 = psi0(x,g_x0,g_k0,g_sig)
psi_t1 = psi_t0
psi_tu05 = []
for t in range(7):
    psi_t1 = Ut_05s.dot(psi_t1)
    psi_tu05.append( (t,psi_t1))
# psi_tu.append( (t,psi_t1))
print "Start"
for t,p in psi_tu05:
    norm = np.linalg.norm(p)
    vev_E1=float(np.real(np.sum(np.conjugate(p)*H.dot(p))))
    vev_x1=float(np.real(np.sum(np.conjugate(p)*x*p)))
    print "dt={:7.1f}  norm = {:8.5f}  <E> = {:8.4f}  <x_(dt)>= {:8.4g}".format(t,norm,vev_E1,vev_x1)


fig = plt.figure(figsize=[10,7])
if vev_E0>max(V):
    plt.title('Scattering')
else:
    plt.title('Tunneling')
ax1 = fig.add_subplot(111)
ax1.set_ylabel('$\psi(x,t)$')
ax1.set_xlabel('$x$')
line, = ax1.plot(x,np.abs(psi_t0)/np.sqrt(Delta_x),label="$\Psi(x,t=0)$")
for t,p in psi_tu05:
    ax1.plot(x,np.abs(p)/np.sqrt(Delta_x),label="$\Psi(x,t={:6.3f})$".format(t*dt))
ax1.legend(loc='upper left')
ax2=ax1.twinx()
ax2.plot(x,V,color="grey",label="V(x)")
ax2.plot([x[0],x[N-1]],[vev_E0,vev_E0],color="grey",linestyle="--",label="<E>")
ax2.set_ylabel("Energy",color="grey")
ax2.set_xlim(g_x0-5*g_sig,-g_x0+5*g_sig)
ax2.legend(loc='upper right')
#plt.savefig("Displaced_state.pdf")
plt.show()

n = np.arange(N)
k0 = -np.pi/Delta_x
kN =  np.pi/Delta_x
Delta_k = (kN-k0)/N
k = k0 + n*Delta_k 

g_x0=-10.
g_k0=6.
g_sig=2.

# Gaussian in k-space, where p = \hbar k, with a momentum k0, and x-space psi(x,0) (2a/pi)^1/4 e^{-ax^2}
def psi0(x,g_x0,g_k0,g_sig):
    _Norm_x=np.sqrt(Delta_x/g_sig)/(np.pi**0.25)
    return(_Norm_x*np.exp(-(x-g_x0)**2/(2.*g_sig*g_sig)+1j*g_k0*x))

psi_t0 = psi0(x,g_x0,g_k0,g_sig)

def Psi_Evolve(dt,psi0):
    '''This function will evolve the wave function by an amount dt and return the new wave function.'''
    #psi_scaled = Delta_x/np.sqrt(2*np.pi) *psi0* np.exp(-1j *k0*x)
    psi_scaled = psi0* np.exp(-1j *k0*x)
    # Take the 1/2 step in x-space with psi_scaled
    psi_scaled *= np.exp(-1j*0.5*V*dt/hbar)
    phi_scaled = fft.fft(psi_scaled)
    # Take the step in k-space
    phi_scaled *= np.exp(-1j*hbar**3*k**2*dt/(2*m))
    psi_scaled = fft.ifft(phi_scaled)
    # Take the 1/2 step in x-space with psi_scaled
    psi_scaled *= np.exp(-1j*0.5*V*dt/hbar)
    #psi_out = psi_scaled*np.sqrt(2*np.pi)/Delta_x * np.exp(+1j *k0*x)
    psi_out = psi_scaled* np.exp(+1j *k0*x)
    return(psi_out)

def Psi_Evolve_N(dt,psi0,N):
    '''This function will evolve the wave function by an amount N*dt and return the new wave function.'''
    psi_scaled = psi0* np.exp(-1j *k0*x)
    # Take the 1/2 step in x-space with psi_scaled
    psi_scaled *= np.exp(-1j*0.5*V*dt/hbar)
    
    for i in range(N-1):     # We now take N-1 times a step in k, and then a step in x.
        phi_scaled = fft.fft(psi_scaled)
        # Take the step in k-space
        phi_scaled *= np.exp(-1j*hbar**3*k**2*dt/(2*m))
        psi_scaled = fft.ifft(phi_scaled)
        psi_scaled *= np.exp(-1j*V*dt/hbar)
 
    phi_scaled = fft.fft(psi_scaled)
    # Take the last step in k-space
    phi_scaled *= np.exp(-1j*hbar**3*k**2*dt/(2*m))
    psi_scaled = fft.ifft(phi_scaled)
    # Take the last 1/2 step in x-space with psi_scaled
    psi_scaled *= np.exp(-1j*0.5*V*dt/hbar)
    #psi_out = psi_scaled*np.sqrt(2*np.pi)/Delta_x * np.exp(+1j *k0*x)
    psi_out = psi_scaled* np.exp(+1j *k0*x)
    return(psi_out)

psi_t0 = psi0(x,g_x0,g_k0,g_sig)
psi_t1 = psi_t0
psi_t = []
for i in range(7):
    t=i*500*dt
    psi_t1 = Psi_Evolve_N(dt,psi_t1,500)
#    for m in range(500):
#        psi_t1 = Psi_Evolve(dt,psi_t1)
    psi_t.append( (t,psi_t1))

print "Normalization: ",np.linalg.norm(psi_t[-1][1])
vev_E0=float(np.real(np.sum(np.conjugate(psi_t0)*H.dot(psi_t0))))
vev_x0=float(np.real(np.sum(np.conjugate(psi_t0)*x*psi_t0)))
print "<E_(t=0)> = {:8.4f}  <x_(t=0)>= {:8.4f}".format(vev_E0,vev_x0)

for t,p in psi_t:
    norm = np.linalg.norm(p)
    vev_E1=float(np.real(np.sum(np.conjugate(p)*H.dot(p))))
    vev_x1=float(np.real(np.sum(np.conjugate(p)*x*p)))
    print "t={:5.2f}  norm = {:8.5f}  <E> = {:8.4f}  <x_(dt)>= {:8.4g}".format(t,norm,vev_E1,vev_x1)

print N

fig,ax1 = plt.subplots(figsize=[10,7])
if vev_E0>max(V):
    plt.title('Scattering')
else:
    plt.title('Tunneling')
ax1.set_ylabel('$\psi(x,t)$')
ax1.set_xlabel('$x$')
#ax1.plot(x,np.abs(psi_t0)/np.sqrt(Delta_x),label="$\Psi(x,t=0)$")
for t,p in psi_t:
    ax1.plot(x,np.abs(p)/np.sqrt(Delta_x),label="$\Psi(x,t={:6.3f})$".format(t))
ax1.legend(loc='upper left')
ax2=ax1.twinx()
ax2.plot(x,V,color="grey",label="V(x)")
ax2.plot([x[0],x[N-1]],[vev_E0,vev_E0],color="grey",linestyle="--",label="<E>")
ax2.set_ylabel("Energy",color="grey")
ax2.set_xlim(g_x0-5*g_sig,-g_x0+5*g_sig)
ax2.legend(loc='upper right')
#plt.savefig("Displaced_state.pdf")
plt.show()

import matplotlib.animation as animation
from IPython.display import HTML

frame_rate = 20     # Frame rate in Hz. Make higher for smoother movie, but it takes longer to compute.
time_slowdown = 2   # Run time x times slower than normal. we want this about 10.
T=7.                # Total time to animate.
x_min = -30.
x_max = +30.

#%time Ut_1f = np.linalg.matrix_power(Ut_mat,int(1./frame_rate/dt) )  # Moves the wave by one frame according to frame rate
print int(1./frame_rate/dt)

psi_t0 = psi0(x,g_x0,g_k0,g_sig)
psi_t1 = psi_t0

fig = plt.figure(figsize=[10,7])
if vev_E0>max(V):
    plt.title('Scattering')
else:
    plt.title('Tunneling')
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.set_ylabel('$\psi(x,t)$')
ax1.set_xlabel('$x$')
ax1.set_xlim(x_min,x_max)
ax1.set_ylim(-0.1,0.9)
line, = ax1.plot([],[],lw=2,color='red')

time_template = 'time = {:9.2f}s'
time_text = ax1.text(0.05, 0.73, '', transform=ax1.transAxes)


def init():
    ax2=ax1.twinx()
    ax2.plot(x,V,color="grey",label="V(x)")
    ax2.plot([x[0],x[N-1]],[vev_E0,vev_E0],color="grey",linestyle="--",label="<E>")
    ax2.set_ylabel("Energy",color="grey")
    ax2.legend(loc='upper right')
    ax1.set_xlim(x_min,x_max)
    ax1.set_ylim(-0.1,0.9)
    line.set_data([], [])
    time_text.set_text(time_template.format(0.))
    return line, time_text

def animate(t):
    global psi_t1
    #psi_t1 = Ut_1f.dot(psi_t1)
    psi_t1 = Psi_Evolve_N(dt,psi_t1,int(1./frame_rate/dt))
    line.set_data(x,np.abs(psi_t1/np.sqrt(Delta_x)))
    time_text.set_text(time_template.format(t))
    return line,  time_text

ani = animation.FuncAnimation(fig, animate, np.linspace(0.,T,int(T*frame_rate)),
                              interval=time_slowdown*1000./frame_rate, blit=True, init_func=init)
HTML(ani.to_html5_video())



