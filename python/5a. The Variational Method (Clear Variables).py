# Import libraries and set up in-line plotting.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
import numpy as np

# Define the x-axis from -width to +width
# This makes the QHO finite in width, which is an approximation
# We will need to be careful not to make omega too small
width = 10.0
num_x_points = 1001
x = np.linspace((-width),width,num_x_points)
dx = 2.0*width/(num_x_points - 1)

# Integrate the product of two functions over the width of the well
# NB this is a VERY simple integration routine: there are much better ways
def integrate_functions(function1,function2,dx):
    """Integrate the product of two functions over defined x range with spacing dx"""
    # We use the NumPy dot function here instead of iterating over array elements
    integral = dx*np.dot(function1,function2)
    return integral

# Define a function to act on a basis function with the potential
def add_potential_on_basis(Hphi,V,phi):
    for i in range(V.size):
        Hphi[i] = Hphi[i] + V[i]*phi[i]

# Define the potential in the square well
def square_well_potential(x,V,a):
    """Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    for i in range(x.size):
        V[i] = 0.5*a*x[i]*x[i]
    # If necessary, plot to ensure that we know what we're getting
    #pl.plot(x,V)
    

# Make an array for the potential and create it before plotting it
omega = 2.0 # Set the frequency
Potential_QHO = np.linspace(0.0,width,num_x_points)
square_well_potential(x,Potential_QHO,omega*omega)
pl.plot(x,Potential_QHO)
pl.xlim(-width,width)

from math import pi
root_pi = np.sqrt(pi)
# Define a gaussian with proper normalisation as our test function
def gaussian(x,alpha):
    return np.sqrt(alpha / (root_pi))* np.exp(-0.5 * alpha**2 * x**2)
# We can also write the second derivative function easily enough
def second_derivative_gaussian(x,alpha):
    return np.sqrt(alpha / (root_pi))* alpha**2 * (alpha**2*x**2 - 1) * np.exp(-0.5 * alpha**2 * x**2)

# Declare space for the potential and call the routine
omega = 2.0 # Set the frequency
Potential_QHO = np.linspace(0.0,width,num_x_points)
square_well_potential(x,Potential_QHO,omega*omega)
print "From theory, E min, alpha: ",0.5*omega,np.sqrt(omega)
psi = gaussian(x,np.sqrt(omega))
# Check that I've normalised the gaussian correctly
print "Norm is: ",integrate_functions(psi,psi,dx)
pl.plot(x,psi)

psi = gaussian(x,np.sqrt(omega))
# Kinetic energy
Hpsi = -0.5*second_derivative_gaussian(x,np.sqrt(omega))
# Add potential
add_potential_on_basis(Hpsi,Potential_QHO,psi)
# Check the exact answer - we won't be able to do this normally !
print "Energy at optimum alpha is: ",integrate_functions(psi,Hpsi,dx)

alpha_values = np.linspace(0.1,10.1,num_x_points)
energy = np.zeros(num_x_points)
i=0
Energy_minimum = 1e30
Alpha_minimum = 0.0
for alpha in alpha_values:
    psi = gaussian(x,alpha)
    norm = integrate_functions(psi,psi,dx)
    #if np.abs(norm-1.0)>1e-6:
    #    print "Norm error: ",alpha,norm
    Hpsi = -0.5*second_derivative_gaussian(x,alpha)
    add_potential_on_basis(Hpsi,Potential_QHO,psi)
    energy[i] = integrate_functions(psi,Hpsi,dx)
    if energy[i]<Energy_minimum:
        Energy_minimum = energy[i]
        Alpha_minimum = alpha
    i=i+1
print "Min E and alph: ",Energy_minimum, Alpha_minimum
pl.plot(alpha_values,energy)
pl.xlabel(r"Value of $\alpha$")
pl.ylabel("Energy")

# Define the eigenbasis - normalisation needed elsewhere
def square_well_eigenbasis(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    fac = np.pi*n/width
    return norm*np.sin(fac*x)

# We will also define the second derivative for kinetic energy (KE)
def second_derivative_square_well_eigenbasis(n,width,norm,x):
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
second_derivative_basis_array = np.zeros((num_basis,num_x_points))

# Loop over first num_basis basis states, normalise and create an array
# NB the basis_array will start from 0
for i in range(num_basis):
    n = i+1
    # Calculate A = <phi_n|phi_n>
    integral = integrate_functions(square_well_eigenbasis(n,width,1.0,x),square_well_eigenbasis(n,width,1.0,x),dx)
    # Use 1/sqrt{A} as normalisation constant
    normalisation = 1.0/np.sqrt(integral)
    basis_array[i,:] = square_well_eigenbasis(n,width,normalisation,x)
    second_derivative_basis_array[i,:] = second_derivative_square_well_eigenbasis(n,width,normalisation,x)

# Define the potential in the square well
def square_well_linear_potential(x,V,a):
    """Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    for i in range(x.size):
        V[i] = a*(x[i]-width/2.0)
    # Plot to ensure that we know what we're getting
    pl.plot(x,V)
    pl.title("Potential")
    
# Declare an array for this potential (Diagonal_Potential) and find the potential's values
Diagonal_Potential = np.linspace(0.0,width,num_x_points)
square_well_linear_potential(x,Diagonal_Potential,1.0)

# Declare space for the matrix elements
H_Matrix2 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_basis_array[m] 
        add_potential_on_basis(H_phi_m,Diagonal_Potential,basis_array[m])
        # Create matrix element by integrating
        H_Matrix2[m,n] = integrate_functions(basis_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_Matrix2[m,n],
    # This print puts in a new line when we have finished looping over m
    print
  

n_alpha = 101
alpha_values = np.linspace(-0.1,0.1,n_alpha)
energy2 = np.zeros(n_alpha)
i=0
Energy_minimum = 1e30
Alpha_minimum = 0.0
for alpha in alpha_values:
    psi = basis_array[0] + alpha*basis_array[1]
    H_psi = -0.5*(second_derivative_basis_array[0] + alpha*second_derivative_basis_array[1])
    add_potential_on_basis(H_psi,Diagonal_Potential,psi)
    norm = integrate_functions(psi,psi,dx)
    #print alpha, norm
    #print H_Matrix2[0,0] + H_Matrix2[1,0]*alpha + H_Matrix2[0,1]*alpha + H_Matrix2[1,1]*alpha*alpha
    energy2[i] = integrate_functions(psi,H_psi,dx)/norm
    if energy2[i]<Energy_minimum:
        Energy_minimum = energy2[i]
        Alpha_minimum = alpha
    i=i+1
print "Minimum Energy and alpha: ",Energy_minimum, Alpha_minimum
pl.plot(alpha_values,energy2)
pl.xlabel(r"Value of $\alpha$")
pl.ylabel("Energy")

Diagonal_Potential2 = np.linspace(0.0,width,num_x_points)
square_well_linear_potential(x,Diagonal_Potential2,20.0) # A much larger potential
# Declare space for the matrix elements
H_Matrix3 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_basis_array[m] 
        add_potential_on_basis(H_phi_m,Diagonal_Potential2,basis_array[m])
        # Create matrix element by integrating
        H_Matrix3[m,n] = integrate_functions(basis_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_Matrix3[m,n],
    # This print puts in a new line when we have finished looping over m
    print

# Use exact algebra to find the result for comparison
import numpy.linalg as la
Energy_values, Energy_vectors = la.eigh(H_Matrix3)
print "Ground state energy: ",Energy_values[0]
print "Ground state wavevector: ",Energy_vectors[:,0]

# Now set up the simple parameter scan
n_alpha = 101
alpha_values = np.linspace(-1,1,n_alpha)
energy3 = np.zeros(n_alpha)
i=0
Energy_minimum = 1e30
Alpha_minimum = 0.0
for alpha in alpha_values:
    psi = basis_array[0] + alpha*basis_array[1]
    H_psi = -0.5*(second_derivative_basis_array[0] + alpha*second_derivative_basis_array[1])
    add_potential_on_basis(H_psi,Diagonal_Potential2,psi)
    norm = integrate_functions(psi,psi,dx)
    #print alpha, norm
    #print H_Matrix2[0,0] + H_Matrix2[1,0]*alpha + H_Matrix2[0,1]*alpha + H_Matrix2[1,1]*alpha*alpha
    energy3[i] = integrate_functions(psi,H_psi,dx)/norm
    if energy3[i]<Energy_minimum:
        Energy_minimum = energy3[i]
        Alpha_minimum = alpha
    i=i+1
print "Minimum Energy and alpha: ",Energy_minimum, Alpha_minimum
pl.plot(alpha_values,energy3)
pl.xlabel(r"Value of $\alpha$")
pl.ylabel("Energy")



