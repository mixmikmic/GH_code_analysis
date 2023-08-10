# Import libraries and set up in-line plotting.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
import numpy as np
# This is a new library - linear algebra includes solving for eigenvalues & eigenvectors of matrices
import numpy.linalg as la

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

# Integrate two functions over the width of the well
def integrate_functions(f1,f2,size_x,dx):
    """Integrate two functions over defined x range"""
    sum = 0.0
    for i in range(size_x):
        sum = sum + f1[i]*f2[i]
    sum = sum*dx
    return sum

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
    
# Define a function to act on a basis function with the potential
def add_pot_on_basis(Hphi,V,phi):
    for i in range(V.size):
        Hphi[i] = Hphi[i] + V[i]*phi[i]

# Define the potential in the square well
def square_well_potential2(x,V,a):
    """QHO Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    # Note that we offset x so that the QHO well is centred on the square well
    for i in range(x.size):
        V[i] = 0.5*a*(x[i]-0.5)**2 
    # Plot to ensure that we know what we're getting
    pl.plot(x,V)
    
omega = 1.0
omega2 = omega*omega
VbumpQHO = np.linspace(0.0,width,num_x_points)
pot_bump = square_well_potential2(x,VbumpQHO,omega2)

# Declare space for the matrix elements
Hmat2 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*d2basis_array[m] 
        add_pot_on_basis(H_phi_m,VbumpQHO,basis_array[m])
        # Create matrix element by integrating
        Hmat2[m,n] = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % Hmat2[m,n],
    # This print puts in a new line when we have finished looping over m
    print
    
print "Perturbation matrix elements:"
# Output the matrix elements of the potential to see how large the perturbation is
# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = np.zeros(num_x_points)
        add_pot_on_basis(H_phi_m,VbumpQHO,basis_array[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print
    
# Solve using linalg module of numpy (which we've imported as la above)
eigval, eigvec = la.eigh(Hmat2)
# This call above does the entire solution for the eigenvalues and eigenvectors !
# Print results roughly, though apply precision of 4 to the printing
print
print "Eigenvalues and eigenvector coefficients printed roughly"
np.set_printoptions(precision=4)
print eigval
print eigvec[0]
print eigvec[1]
print eigvec[2]
print

print " QHO Square  Perfect QHO  Difference  Perfect Square Difference"
for i in range(num_basis):
    n = i+1
    print "   %8.3f     %8.3f    %8.3f        %8.3f   %8.3f" % (eigval[i],omega*(i+0.5),eigval[i] - omega*(i+0.5),n*n*np.pi*np.pi/2.0,eigval[i] - n*n*np.pi*np.pi/2.0)

from scipy.special import hermite
from scipy.misc import factorial
from math import pi
root_pi = np.sqrt(pi)
def N(n, alpha):
    return np.sqrt(alpha / (root_pi * (2.0**n) * factorial(n)))
def phi(x,n,alpha):
    return N(n,alpha) * hermite(n)(alpha * x) * np.exp(-0.5 * alpha**2 * x**2)

x2 = np.linspace(-width/2,width/2,num_x_points)
#pl.plot(x,phi(x2,1,np.sqrt(np.sqrt(omega))))

# Define a figure to take two plots
fig3 = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
axb = fig3.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2.1,2.1))
axb.set_title("Eigenvectors for perturbed system")
axb2 = fig3.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-0.001,0.001))
axb2.set_title("Difference to perfect square well eigenvectors")
#axb2.set_title("Difference to QHO eigenvectors")
for m in range(3): # Plot the first four states
    psi = np.zeros(num_x_points)
    for i in range(num_basis):
        psi = psi+eigvec[i,m]*basis_array[i]
    if 2*(m/2)!=m:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    axb.plot(x,psi)
    psi = psi - basis_array[m]
    axb2.plot(x,psi)

# Make omega larger, so that the QHO energy dominates the square well
omegaL = 50.0
omegaL2 = omegaL*omegaL
VbumpQHO2 = np.linspace(0.0,width,num_x_points)
pot_bump = square_well_potential2(x,VbumpQHO2,omegaL2)

# Declare space for the matrix elements
Hmat3 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*d2basis_array[m] 
        add_pot_on_basis(H_phi_m,VbumpQHO2,basis_array[m])
        # Create matrix element by integrating
        Hmat3[m,n] = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % Hmat3[m,n],
    # This print puts in a new line when we have finished looping over m
    print
    
print "Perturbation matrix elements:"
# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = np.zeros(num_x_points)
        add_pot_on_basis(H_phi_m,VbumpQHO2,basis_array[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print
    
# Solve using linalg module of numpy (which we've imported as la above)
eigval, eigvec = la.eigh(Hmat3)
# This call above does the entire solution for the eigenvalues and eigenvectors !
# Print results roughly, though apply precision of 4 to the printing
print
print "Eigenvalues and eigenvector coefficients printed roughly"
np.set_printoptions(precision=4)
print eigval
print eigvec[0]
print eigvec[1]
print eigvec[2]
print

print " QHO Square  Perfect QHO  Difference  Perfect Square Difference"
for i in range(num_basis):
    n = i+1
    print "   %8.3f     %8.3f    %8.3f        %8.3f   %8.3f" % (eigval[i],omegaL*(i+0.5),eigval[i] - omegaL*(i+0.5),n*n*np.pi*np.pi/2.0,eigval[i] - n*n*np.pi*np.pi/2.0)

# Define a figure to take two plots
fig4 = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
axc = fig4.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2.1,2.1))
axc.set_title("Eigenvectors for perturbed system")
axc2 = fig4.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-0.1,0.1))
#axc2.set_title("QHO eigenvectors")
axc2.set_title("Difference to QHO eigenvectors")
for m in range(3): # Plot the first four states
    psi = np.zeros(num_x_points)
    for i in range(num_basis):
        psi = psi+eigvec[i,m]*basis_array[i]
    #if 2*(m/2)!=m:  # This is just to ensure that psi and the basis function have the same phase
    #    psi = -psi
    axc.plot(x,psi)
    psi = psi - phi(x2,m,np.sqrt(omegaL))
    axc2.plot(x,psi)

# Make omega larger, so that the QHO energy dominates the square well
omegaH = 500.0
omegaH2 = omegaH*omegaH
VbumpQHO3 = np.linspace(0.0,width,num_x_points)
pot_bump = square_well_potential2(x,VbumpQHO3,omegaH2)

# Declare space for the matrix elements
Hmat4 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*d2basis_array[m] 
        add_pot_on_basis(H_phi_m,VbumpQHO3,basis_array[m])
        # Create matrix element by integrating
        Hmat4[m,n] = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % Hmat4[m,n],
    # This print puts in a new line when we have finished looping over m
    print
    
# Solve using linalg module of numpy (which we've imported as la above)
eigval, eigvec = la.eigh(Hmat4)
print " QHO Square  Perfect QHO  Difference"
for i in range(num_basis):
    n = i+1
    print "   %8.3f     %8.3f    %8.3f" % (eigval[i],omegaH*(i+0.5),eigval[i] - omegaH*(i+0.5))

# Define a figure to take two plots
fig5 = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
axd = fig5.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-3.1,3.1))
axd.set_title("Eigenvectors for perturbed system")
axd2 = fig5.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-3.1,3.1))
#axc2.set_title("QHO eigenvectors")
axd2.set_title("QHO eigenvectors")
for m in range(3): # Plot the first four states
    psi = np.zeros(num_x_points)
    for i in range(num_basis):
        psi = psi+eigvec[i,m]*basis_array[i]
    if 2*(m/2)!=m:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    axd.plot(x,psi)
    #psi = psi - phi(x2,m,np.sqrt(omegaH))
    #axd2.plot(x,psi)
    axd2.plot(x,phi(x2,m,np.sqrt(omegaH)))



