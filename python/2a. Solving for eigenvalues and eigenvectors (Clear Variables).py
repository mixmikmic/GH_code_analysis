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
square_well_width = 1.0
number_of_x_points = 101
x = np.linspace(0.0,square_well_width,number_of_x_points)
x_spacing = square_well_width/(number_of_x_points - 1)

# Integrate two functions over the width of the well
# NB this is a VERY simple integration routine: there are much better ways
def integrate_functions(function1,function2,dx):
    """Integrate the product of two functions over defined x range with spacing dx"""
    # We use the NumPy dot function here instead of iterating over array elements
    integral = dx*np.dot(function1,function2)
    return integral

# These arrays will each hold an array of functions
basis_functions = []
second_derivative_basis_functions = []

number_of_basis_functions = 10
# Loop over first num_basis basis states, normalise and create an array
# NB the basis_array will start from 0
for n in range(1,number_of_basis_functions+1):
    # Calculate A = <phi_n|phi_n>
    integral = integrate_functions(square_well_eigenfunctions(n,square_well_width,1.0,x),square_well_eigenfunctions(n,square_well_width,1.0,x),x_spacing)
    # Use 1/sqrt{A} as normalisation constant
    normalisation = 1.0/np.sqrt(integral)
    basis_functions.append(square_well_eigenfunctions(n,square_well_width,normalisation,x))
    second_derivative_basis_functions.append(second_derivative_square_well_eigenfunctions(n,square_well_width,normalisation,x))

# Define constants - here I use atomic units
hbar = 1.0
hbar_squared = hbar**2
m_e = 1.0 # Mass of the electron
two_m_e = 2.0*m_e
# These are not needed but are easier
pi_squared = np.pi**2
square_well_width_squared = square_well_width**2

# Define the potential in the square well
def square_well_perturbing_potential(x,perturbing_potential,potential_magnitude):
    """Potential for a particle in a square well, expecting two arrays: x, V(x), and potential magnitude"""
    # Zero the array
    perturbing_potential[:] = 0.0
    # Let's define a small bump in the well
    bump_position = x.size/4
    perturbing_potential[bump_position-1] = potential_magnitude
    perturbing_potential[bump_position] = potential_magnitude
    perturbing_potential[bump_position+1] = potential_magnitude
    # Plot to ensure that we know what we're getting
    pl.plot(x,perturbing_potential)
    pl.ylim((0,2.0))
    
# Declare space for this potential (Vbump) and call the routine
bump_potential = np.linspace(0.0,square_well_width,number_of_x_points)
square_well_perturbing_potential(x,bump_potential,0.5)

# Declare space for the matrix elements - simplest with the identity function
H_matrix = np.eye(number_of_basis_functions)

# Define a function to act on a basis function with the potential
def add_potential_acting_on_state(H_on_phi,V,phi):
    """Add V(x)phi(x) onto an input array, H_on_phi"""
    for i in range(V.size):
        H_on_phi[i] = H_on_phi[i] + V[i] * phi[i]
        
print "Potential matrix elements:"
# Loop over basis functions phi_n (the bra in the matrix element)
# Calculate and output matrix elements of the potential

for n in range(number_of_basis_functions):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(number_of_basis_functions):
        # Act with H on phi_m and store in H_phi_m
        H_on_phi_m = np.zeros(number_of_x_points)
        add_potential_acting_on_state(H_on_phi_m, bump_potential, basis_functions[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_functions[n],H_on_phi_m,x_spacing)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print

print
print "Full Hamiltonian"
# Loop over basis functions phi_n (the bra in the matrix element)
# Calculate and store the matrix elements for the full Hamiltonian
for n in range(number_of_basis_functions):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(number_of_basis_functions):
        # Act with H on phi_m and store in H_phi_m
        # First the kinetic energy
        H_on_phi_m = -(hbar_squared / two_m_e) * second_derivative_basis_functions[m] 
        # Now the potential
        add_potential_acting_on_state(H_on_phi_m, bump_potential, basis_functions[m])
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_functions[n], H_on_phi_m, x_spacing)
        H_matrix[m,n] = H_mn
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print

# Solve using linalg module of numpy (which we've imported as la above)
eigenvalues, eigenvectors = la.eigh(H_matrix)
# This call above does the entire solution for the eigenvalues and eigenvectors !
# Print results roughly, though apply precision of 4 to the printing
np.set_printoptions(precision=4)
print eigenvalues
print eigenvectors[0]
print eigenvectors[1]
print eigenvectors[2]

# Now print out eigenvalues and the eigenvalues of the perfect square well, and the difference
print " Changed Original  Difference"
for i in range(number_of_basis_functions):
    n = i+1
    print "%8.3f %8.3f %8.3f" % (eigenvalues[i],n*n*np.pi*np.pi/2.0,eigenvalues[i] - n*n*np.pi*np.pi/2.0)

# Define a figure to take two plots
fig = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
ax = fig.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2,2))
ax.set_title("Eigenvectors for changed system")
ax2 = fig.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-0.004,0.004))
ax2.set_title("Difference to perfect eigenvectors")
for m in range(4): # Plot the first four states
    psi = np.zeros(number_of_x_points)
    for i in range(number_of_basis_functions):
        psi = psi+eigenvectors[i,m]*basis_functions[i]
    if eigenvectors[m,m] < 0:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    ax.plot(x,psi)
    # Now subtract the unperturbed eigenvector to see the change from the perturbation
    psi = psi - basis_functions[m]
    ax2.plot(x,psi)



