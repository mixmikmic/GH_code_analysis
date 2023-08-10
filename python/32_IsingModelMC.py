import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

def initialize_config(L):
    '''Initialize a random spin configuration'''

    σ = np.ones([L,L],dtype=int)
    σ[np.random.random([L,L])<=0.5] = -1
    return σ


L = 10
σ = initialize_config(L)

# visualize
plt.matshow(σ, cmap='gray', extent=[0,9,0,9]);
plt.xticks([]);
plt.yticks([]);

p1 = np.arange(1,L+1)
p1[-1] = 0
m1 = np.arange(-1,L-1)
m1[0] - L-1

print(p1)
print(m1)

def get_props(σ):
    '''The energy E and magnetization M for a microstate of the 2d Ising model.'''
    E,M = 0,0

    return E,M

def monte_carlo_step(σ,T):
    '''Perform a Monte Carlo step.'''
    
    # get the current magnetization
    M = np.sum(σ)
    
    # attempt L^2 spin flips
    for update in range(σ.size):
        
        # get the random spin
        k = np.random.randint(0,σ.shape[0])
        ℓ = np.random.randint(0,σ.shape[1])
        
        # calculate the change in energy
        ΔE = 2*σ[k,ℓ]*(σ[p1[k],ℓ] + σ[m1[k],ℓ] + σ[k,p1[ℓ]] + σ[k,m1[ℓ]])
        
        # perform the Metropolis test
        if np.random.random() <= np.exp(-ΔE/T):
            σ[k,ℓ] *= -1

            # Update the magnetization
            M += 2*σ[k,ℓ]
    
    return M

# temperatures to consider
T = np.arange(0.2,2.5,0.2)

# number of Monte Carlo steps we will perform
num_steps = 10000

# magnetization for each temperature
M = np.zeros([num_steps,T.size])

# initialize
L = 5
σ = initialize_config(L)

# create PBC lookup tables
p1 = np.arange(1,L+1)
p1[-1] = 0
m1 = np.arange(-1,L-1)
m1[0] - L-1

# Loop over temperatures from high to low
for iT,cT in enumerate(T[::-1]):
    m = T.size - 1 - iT
    
    # initialize the magnetization
    M[0,m] = np.sum(σ)
    
    # Perform the Monte Carlo steps
    for step in range(1,num_steps):
        M[step,m] = monte_carlo_step(σ,cT)

iT = 8
plt.plot(M[:,iT]/L**2,'-', label='T = %3.1f' %T[iT])
plt.legend()
plt.xlabel('MC Step')
plt.ylabel('Magnetization per spin')

skip = 2000
m = np.average(M[skip:]/L**2,axis=0)
δm = np.std(M[skip:]/L**2,axis=0)/np.sqrt(num_steps-skip)

plt.errorbar(T,np.abs(m),yerr=δm, linewidth=0.5, marker='o', markerfacecolor='None', markeredgecolor=colors[0], markersize=4, elinewidth=0.5)
plt.axvline(x=2.0/np.log(1.0+np.sqrt(2.0)), linewidth=1, color='gray', linestyle='--')
plt.xlim(0.2,2.4)
plt.ylim(-0.01,1.1)
plt.xlabel(r'Temperature ($k_{\rm B}T/J$)')
plt.ylabel('Magnetization per spin')
plt.title("L = 5")

def magnetization_exact_(T):
    '''We use units where J/k_B = 1.'''
    Tc = 2.0/np.log(1.0+np.sqrt(2.0))
    if T < Tc:
        return (1.0 - np.sinh(2.0/T)**(-4))**(1.0/8)
    else:
        return 0.0
magnetization_exact = np.vectorize(magnetization_exact_)

data = np.loadtxt('data/Ising_estimators_032.dat')
lT = np.linspace(0.01,4,1000)
lL = 32

plt.plot(lT,magnetization_exact(lT),'-k', linewidth=1, label='Exact')
plt.errorbar(data[:,0],np.abs(data[:,5])/lL**2,yerr=data[:,6]/lL**2, linewidth=0.5, marker='o', markerfacecolor='None', 
             markeredgecolor=colors[0], markersize=4, elinewidth=0.5, label='Monte Carlo')
plt.xlim(0.2,4)
plt.ylim(0,1.1)
plt.xlabel(r'Temperature ($k_{\rm B}T/J$)')
plt.ylabel('Magnetization per spin')
plt.title("L = 32")
plt.legend()

