# Import libraries and set up in-line plotting.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
import numpy as np
# This is a new library - linear algebra includes solving for eigenvalues & eigenvectors of matrices
import numpy.linalg as la

# Define the eigenbasis - normalisation needed elsewhere
def square_well_eigenfunctions(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    fac = np.pi*n/width
    return norm*np.sin(fac*x)

# We will also define the second derivative for kinetic energy (KE)
def second_derivative_square_well_eigenfunctions(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    fac = np.pi*n/width
    return -fac*fac*norm*np.sin(fac*x)

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
second_derivative_functions_array = np.zeros((num_basis,num_x_points))

# Loop over first num_basis basis states, normalise and create an array
# NB the basis_functions_array will start from 0
for i in range(num_basis):
    n = i+1
    # Calculate A = <phi_n|phi_n>
    integral = integrate_functions(square_well_eigenfunctions(n,width,1.0,x),square_well_eigenfunctions(n,width,1.0,x),dx)
    # Use 1/sqrt{A} as normalisation constant
    normalisation = 1.0/np.sqrt(integral)
    basis_functions_array[i,:] = square_well_eigenfunctions(n,width,normalisation,x)
    second_derivative_functions_array[i,:] = second_derivative_square_well_eigenfunctions(n,width,normalisation,x)
    
# Define a function to act on a basis function with the potential
def add_potential_on_basis(Hphi,V,phi):
    for i in range(V.size):
        Hphi[i] = Hphi[i] + V[i]*phi[i]

# Define the potential in the square well
def square_well_potential(x,V,a):
    """Potential for a particle in a square well, expecting two arrays: x, V(x), and potential height, a"""
    for i in range(x.size):
        V[i] = a*(x[i]-width/2.0)
    # Plot to ensure that we know what we're getting
    pl.plot(x,V)
    
# Declare space for this potential (diagonal_Potential) and call the routine
diagonal_Potential = np.linspace(0.0,width,num_x_points)
square_well_potential(x,diagonal_Potential,1.0)

# Declare space for the matrix elements
H_matrix2 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_functions_array[m] 
        add_potential_on_basis(H_phi_m,diagonal_Potential,basis_functions_array[m])
        # Create matrix element by integrating
        H_matrix2[m,n] = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_matrix2[m,n],
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
        add_potential_on_basis(H_phi_m,diagonal_Potential,basis_functions_array[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print
    

# Solve using linalg module of numpy (which we've imported as la above)
eigenvalues, eigenvectors = la.eigh(H_matrix2)
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

print "    Diag  Perf Square  Difference"
for i in range(num_basis):
    n = i+1
    print "%8.3f     %8.3f    %8.3f" % (eigenvalues[i],n*n*np.pi*np.pi/2.0,eigenvalues[i] - n*n*np.pi*np.pi/2.0)

# Define a figure to take two plots
fig = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
ax = fig.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2,2))
ax.set_title("Eigenvectors for changed system")
ax2 = fig.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-0.05,0.05))
ax2.set_title("Difference to perfect eigenvectors")
for m in range(4): # Plot the first four states
    psi = np.zeros(num_x_points)
    for i in range(num_basis):
        psi = psi+eigenvectors[i,m]*basis_functions_array[i]
    if eigenvectors[m,m] < 0:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    ax.plot(x,psi)
    psi = psi - basis_functions_array[m]
    ax2.plot(x,psi)

diagonal_Potential2 = np.linspace(0.0,width,num_x_points)
square_well_potential(x,diagonal_Potential2,20.0)
# Declare space for the matrix elements
H_matrix3 = np.eye(num_basis)

# Loop over basis functions phi_n (the bra in the matrix element)
print "Full Hamiltonian"
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        H_phi_m = -0.5*second_derivative_functions_array[m] 
        add_potential_on_basis(H_phi_m,diagonal_Potential2,basis_functions_array[m])
        # Create matrix element by integrating
        H_matrix3[m,n] = integrate_functions(basis_functions_array[n],H_phi_m,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_matrix3[m,n],
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
        add_potential_on_basis(H_phi_m,diagonal_Potential2,basis_functions_array[m])
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

print "    Diag  Perf Square  Difference"
for i in range(num_basis):
    n = i+1
    print "%8.3f     %8.3f    %8.3f" % (eigenvalues[i],n*n*np.pi*np.pi/2.0,eigenvalues[i] - n*n*np.pi*np.pi/2.0)

# Define a figure to take two plots
fig = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
ax = fig.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2,2))
ax.set_title("Eigenvectors for changed system")
ax2 = fig.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-1,1))
ax2.set_title("Difference to perfect eigenvectors")
for m in range(4): # Plot the first four states
    psi = np.zeros(num_x_points)
    for i in range(num_basis):
        psi = psi+eigenvectors[i,m]*basis_functions_array[i]
    if eigenvectors[m,m] < 0:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    ax.plot(x,psi)
    psi = psi - basis_functions_array[m]
    ax2.plot(x,psi)



