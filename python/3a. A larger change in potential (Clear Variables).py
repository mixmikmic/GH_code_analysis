# Import libraries and set up in-line plotting.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
import numpy as np
# This is a new library - linear algebra includes solving for eigenvalues & eigenvectors of matrices
import numpy.linalg as la

# Define the eigenbasis - normalisation needed elsewhere
def square_well_eigenfunctions(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a (width), sin(n pi x/a). 
    N.B. requires a normalisation factor, norm."""
    wavevector = np.pi*n/width
    return norm*np.sin(wavevector*x)

# We will also define the second derivative for kinetic energy (KE)
def second_derivative_square_well_eigenfunctions(n,width,norm,x):
    """The second derivative of the eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    wavevector = np.pi*n/width
    return -wavevector*wavevector*norm*np.sin(wavevector*x)

# Define the x-axis
width = 1.0
num_x_points = 101
x = np.linspace(0.0,width,num_x_points)
dx = width/(num_x_points - 1)

# Integrate the product of two functions over the width of the well
# NB this is a VERY simple integration routine: there are much better ways
def integrate_functions(function1,function2,dx):
    """Integrate the product of two functions over defined x range with spacing dx"""
    # We use the NumPy dot function here instead of iterating over array elements
    integral = dx*np.dot(function1,function2)
    return integral

# Now set up the array of basis functions - specify the size of the basis
num_basis = 10
# These arrays will each hold an array of functions
basis_functions_array = np.zeros((num_basis,num_x_points))
second_derivative_basis_functions_array = np.zeros((num_basis,num_x_points))

# Loop over first num_basis basis states, normalise and create an array
# NB the basis_functions_array will start from 0
for i in range(num_basis):
    n = i+1
    # Calculate A = <phi_n|phi_n>
    integral = integrate_functions(square_well_eigenfunctions(n,width,1.0,x),square_well_eigenfunctions(n,width,1.0,x),dx)
    # Use 1/sqrt{A} as normalisation constant
    normalisation = 1.0/np.sqrt(integral)
    basis_functions_array[i,:] = square_well_eigenfunctions(n,width,normalisation,x)
    second_derivative_basis_functions_array[i,:] = second_derivative_square_well_eigenfunctions(n,width,normalisation,x)
    
# Define a function to act on a basis function with the potential
def add_potential_on_basis(Hphi,V,phi):
    for i in range(V.size):
        Hphi[i] = Hphi[i] + V[i]*phi[i]

# Define the potential in the square well
def square_well_and_QHO_potential(x,V,a):
    """QHO Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    # Note that we offset x so that the QHO well is centred on the square well
    for i in range(x.size):
        V[i] = 0.5*a*(x[i]-0.5)**2 
    # Plot to ensure that we know what we're getting
    pl.plot(x,V)
    
omega = 1.0
omega_2 = omega**2
Potential_with_QHO = np.linspace(0.0,width,num_x_points)
potential_bump = square_well_and_QHO_potential(x,Potential_with_QHO,omega_2)

# Declare space for the matrix elements
H_matrix = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_basis_functions_array[m] 
        add_potential_on_basis(H_phi_m,Potential_with_QHO,basis_functions_array[m])
        # Create matrix element by integrating
        H_matrix[m,n] = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_matrix[m,n],
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
        add_potential_on_basis(H_phi_m,Potential_with_QHO,basis_functions_array[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print
    
# Solve using linalg module of numpy (which we've imported as la above)
eigenvalues, eigenvectors = la.eigh(H_matrix)
# This call above does the entire solution for the eigenvalues and eigenvectors !
# Print results roughly, though apply precision of 4 to the printing
print
print "Eigenvalues and eigenvector coefficients printed roughly"
np.set_printoptions(precision=4)
print eigenvalues
print eigenvectors[0]
print eigenvectors[1]
print eigenvectors[2]
print

print " QHO Square  Perfect QHO  Difference  Perfect Square Difference"
for i in range(num_basis):
    n = i+1
    print "   %8.3f     %8.3f    %8.3f        %8.3f   %8.3f" % (eigenvalues[i],omega*(i+0.5),eigenvalues[i] - omega*(i+0.5),n*n*np.pi*np.pi/2.0,eigenvalues[i] - n*n*np.pi*np.pi/2.0)

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
        psi = psi+eigenvectors[i,m]*basis_functions_array[i]
    if 2*(m/2)!=m:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    axb.plot(x,psi)
    psi = psi - basis_functions_array[m]
    axb2.plot(x,psi)

# Make omega larger, so that the QHO energy dominates the square well
omegaLarger = 50.0
omegaLarger2 = omegaLarger**2
Potential_with_QHO2 = np.linspace(0.0,width,num_x_points)
potential_bump = square_well_and_QHO_potential(x,Potential_with_QHO2,omegaLarger2)

# Declare space for the matrix elements
H_matrix3 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_basis_functions_array[m] 
        add_potential_on_basis(H_phi_m,Potential_with_QHO2,basis_functions_array[m])
        # Create matrix element by integrating
        H_matrix3[m,n] = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_matrix3[m,n],
    # This print puts in a new line when we have finished looping over m
    print
    
print "Perturbation matrix elements:"
# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = np.zeros(num_x_points)
        add_potential_on_basis(H_phi_m,Potential_with_QHO2,basis_functions_array[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print
    
# Solve using linalg module of numpy (which we've imported as la above)
eigenvalues, eigenvectors = la.eigh(H_matrix3)
# This call above does the entire solution for the eigenvalues and eigenvectors !
# Print results roughly, though apply precision of 4 to the printing
print
print "Eigenvalues and eigenvector coefficients printed roughly"
np.set_printoptions(precision=4)
print eigenvalues
print eigenvectors[0]
print eigenvectors[1]
print eigenvectors[2]
print

print " QHO Square  Perfect QHO  Difference  Perfect Square Difference"
for i in range(num_basis):
    n = i+1
    print "   %8.3f     %8.3f    %8.3f        %8.3f   %8.3f" % (eigenvalues[i],omegaLarger*(i+0.5),eigenvalues[i] - omegaLarger*(i+0.5),n*n*np.pi*np.pi/2.0,eigenvalues[i] - n*n*np.pi*np.pi/2.0)

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
        psi = psi+eigenvectors[i,m]*basis_functions_array[i]
    #if 2*(m/2)!=m:  # This is just to ensure that psi and the basis function have the same phase
    #    psi = -psi
    axc.plot(x,psi)
    psi = psi - phi(x2,m,np.sqrt(omegaLarger))
    axc2.plot(x,psi)

# Make omega larger, so that the QHO energy dominates the square well
omegaH = 500.0
omegaH2 = omegaH**2
Potential_with_QHO3 = np.linspace(0.0,width,num_x_points)
potential_bump = square_well_and_QHO_potential(x,Potential_with_QHO3,omegaH2)

# Declare space for the matrix elements
H_matrix4 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_basis_functions_array[m] 
        add_potential_on_basis(H_phi_m,Potential_with_QHO3,basis_functions_array[m])
        # Create matrix element by integrating
        H_matrix4[m,n] = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_matrix4[m,n],
    # This print puts in a new line when we have finished looping over m
    print
    
# Solve using linalg module of numpy (which we've imported as la above)
eigenvalues, eigenvectors = la.eigh(H_matrix4)
print " QHO Square  Perfect QHO  Difference"
for i in range(num_basis):
    n = i+1
    print "   %8.3f     %8.3f    %8.3f" % (eigenvalues[i],omegaH*(i+0.5),eigenvalues[i] - omegaH*(i+0.5))

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
        psi = psi+eigenvectors[i,m]*basis_functions_array[i]
    if 2*(m/2)!=m:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    axd.plot(x,psi)
    #psi = psi - phi(x2,m,np.sqrt(omegaH))
    #axd2.plot(x,psi)
    axd2.plot(x,phi(x2,m,np.sqrt(omegaH)))



