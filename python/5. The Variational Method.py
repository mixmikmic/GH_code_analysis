# Import libraries and set up in-line plotting.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
import numpy as np

# Define the x-axis from -width to +width
# This makes the QHO finite in width, which is an approximation
# We will need to be careful not to make omega too small
width = 10.0
num_x_points = 1001
x = np.linspace(-width,width,num_x_points)
dx = 2.0*width/(num_x_points - 1)

# Integrate two functions over the width of the well
def integrate_functions(f1,f2,size_x,dx):
    """Integrate two functions over defined x range"""
    sum = 0.0
    for i in range(size_x):
        sum = sum + f1[i]*f2[i]
    sum = sum*dx
    return sum

# Define a function to act on a basis function with the potential
def add_pot_on_basis(Hphi,V,phi):
    for i in range(V.size):
        Hphi[i] = Hphi[i] + V[i]*phi[i]

# Define the potential in the square well
def square_well_potential(x,V,a):
    """Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    for i in range(x.size):
        V[i] = 0.5*a*x[i]*x[i]
    # If necessary, plot to ensure that we know what we're getting
    #pl.plot(x,V)
    

from math import pi
root_pi = np.sqrt(pi)
# Define a gaussian with proper normalisation as our test function
def gauss(x,alpha):
    return np.sqrt(alpha / (root_pi))* np.exp(-0.5 * alpha**2 * x**2)
# We can also write the second derivative function easily enough
def d2gauss(x,alpha):
    return np.sqrt(alpha / (root_pi))* alpha*alpha * (alpha*alpha*x*x - 1) * np.exp(-0.5 * alpha**2 * x**2)

# Declare space for the potential and call the routine
omega = 2.0 # Set the frequency
VQHO = np.linspace(0.0,width,num_x_points)
square_well_potential(x,VQHO,omega*omega)
print "From theory, E min, alpha: ",0.5*omega,np.sqrt(omega)
psi = gauss(x,np.sqrt(omega))
# Check that I've normalised the gaussian correctly
print "Norm is: ",integrate_functions(psi,psi,num_x_points,dx)
pl.plot(x,psi)

psi = gauss(x,np.sqrt(omega))
# Kinetic energy
Hpsi = -0.5*d2gauss(x,np.sqrt(omega))
# Add potential
add_pot_on_basis(Hpsi,VQHO,psi)
# Check the exact answer - we won't be able to do this normally !
print "Energy at optimum alpha is: ",integrate_functions(psi,Hpsi,num_x_points,dx)

alpha_vals = np.linspace(0.1,10.1,1001)
energy = np.zeros(1001)
i=0
e_min = 1e30
alph_min = 0.0
for alpha in alpha_vals:
    psi = gauss(x,alpha)
    norm = integrate_functions(psi,psi,num_x_points,dx)
    #if np.abs(norm-1.0)>1e-6:
    #    print "Norm error: ",alpha,norm
    Hpsi = -0.5*d2gauss(x,alpha)
    add_pot_on_basis(Hpsi,VQHO,psi)
    energy[i] = integrate_functions(psi,Hpsi,num_x_points,dx)
    if energy[i]<e_min:
        e_min = energy[i]
        alpha_min = alpha
    i=i+1
print "Min E and alph: ",e_min, alpha_min
pl.plot(alpha_vals,energy)
pl.xlabel(r"Value of $\alpha$")
pl.ylabel("Energy")

# Define the eigenbasis - normalisation needed elsewhere
def eigenbasis_sw(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    fac = np.pi*n/width
    return norm*np.sin(fac*x)

# We will also define the second derivative for kinetic energy (KE)
def d2eigenbasis_sw(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    fac = np.pi*n/width
    return -fac*fac*norm*np.sin(fac*x)

# Define the x-axis
width = 1.0
num_x_points = 101
x = np.linspace(0.0,width,num_x_points)
dx = width/(num_x_points - 1)
# Now set up the array of basis functions - specify the size of the basis
num_basis = 10
# These arrays will each hold an array of functions
basis_array = np.zeros((num_basis,num_x_points))
d2basis_array = np.zeros((num_basis,num_x_points))

# Loop over first num_basis basis states, normalise and create an array
# NB the basis_array will start from 0
for i in range(num_basis):
    n = i+1
    # Calculate A = <phi_n|phi_n>
    integral = integrate_functions(eigenbasis_sw(n,width,1.0,x),eigenbasis_sw(n,width,1.0,x),num_x_points,dx)
    # Use 1/sqrt{A} as normalisation constant
    normalisation = 1.0/np.sqrt(integral)
    basis_array[i,:] = eigenbasis_sw(n,width,normalisation,x)
    d2basis_array[i,:] = d2eigenbasis_sw(n,width,normalisation,x)

# Define the potential in the square well
def square_well_linear_potential(x,V,a):
    """Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    for i in range(x.size):
        V[i] = a*(x[i]-width/2.0)
    # Plot to ensure that we know what we're getting
    pl.plot(x,V)
    
# Declare space for this potential (Vdiag) and call the routine
Vdiag = np.linspace(0.0,width,num_x_points)
square_well_linear_potential(x,Vdiag,1.0)

# Declare space for the matrix elements
Hmat2 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*d2basis_array[m] 
        add_pot_on_basis(H_phi_m,Vdiag,basis_array[m])
        # Create matrix element by integrating
        Hmat2[m,n] = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % Hmat2[m,n],
    # This print puts in a new line when we have finished looping over m
    print
  

n_alpha = 101
alpha_vals = np.linspace(-0.1,0.1,n_alpha)
energy2 = np.zeros(n_alpha)
i=0
e_min = 1e30
alph_min = 0.0
for alpha in alpha_vals:
    psi = basis_array[0] + alpha*basis_array[1]
    H_psi = -0.5*(d2basis_array[0] + alpha*d2basis_array[1])
    add_pot_on_basis(H_psi,Vdiag,psi)
    norm = integrate_functions(psi,psi,num_x_points,dx)
    #print alpha, norm
    #print Hmat2[0,0] + Hmat2[1,0]*alpha + Hmat2[0,1]*alpha + Hmat2[1,1]*alpha*alpha
    energy2[i] = integrate_functions(psi,H_psi,num_x_points,dx)/norm
    if energy2[i]<e_min:
        e_min = energy2[i]
        alpha_min = alpha
    i=i+1
print "Min E and alph: ",e_min, alpha_min
pl.plot(alpha_vals,energy2)
pl.xlabel(r"Value of $\alpha$")
pl.ylabel("Energy")

Vdiag2 = np.linspace(0.0,width,num_x_points)
square_well_linear_potential(x,Vdiag2,20.0) # A much larger potential
# Declare space for the matrix elements
Hmat3 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*d2basis_array[m] 
        add_pot_on_basis(H_phi_m,Vdiag2,basis_array[m])
        # Create matrix element by integrating
        Hmat3[m,n] = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % Hmat3[m,n],
    # This print puts in a new line when we have finished looping over m
    print

# Use exact algebra to find the result for comparison
import numpy.linalg as la
evals, evecs = la.eigh(Hmat3)
print "Ground state energy: ",evals[0]
print "Ground state wavevector: ",evecs[:,0]

# Now set up the simple parameter scan
n_alpha = 101
alpha_vals = np.linspace(-1,1,n_alpha)
energy3 = np.zeros(n_alpha)
i=0
e_min = 1e30
alph_min = 0.0
for alpha in alpha_vals:
    psi = basis_array[0] + alpha*basis_array[1]
    H_psi = -0.5*(d2basis_array[0] + alpha*d2basis_array[1])
    add_pot_on_basis(H_psi,Vdiag2,psi)
    norm = integrate_functions(psi,psi,num_x_points,dx)
    #print alpha, norm
    #print Hmat2[0,0] + Hmat2[1,0]*alpha + Hmat2[0,1]*alpha + Hmat2[1,1]*alpha*alpha
    energy3[i] = integrate_functions(psi,H_psi,num_x_points,dx)/norm
    if energy3[i]<e_min:
        e_min = energy3[i]
        alpha_min = alpha
    i=i+1
print "Min E and alph: ",e_min, alpha_min
pl.plot(alpha_vals,energy3)
pl.xlabel(r"Value of $\alpha$")
pl.ylabel("Energy")



