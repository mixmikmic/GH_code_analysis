# Import libraries and set up in-line plotting.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
import numpy as np
# This is a new library - linear algebra includes solving for eigenvalues & eigenvectors of matrices
import numpy.linalg as la

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



# Test polynomials
# Now set up the array of basis functions - specify the size of the basis
num_basis = 10
# These arrays will each hold an array of functions
basis_array = np.zeros((num_basis,num_x_points))
d2basis_array = np.zeros((num_basis,num_x_points))

def poly(x,n,width):
    func = x*(width-x)
    print "n is ",n
    for i in range(n):
        print "zero point is ",(i+1.0)/(n+1.0)
        func = func*(x-width*((i+1.0)/(n+1.0)))
    return func

def d2_finite(f,dx):
    d2 = np.zeros(f.size)
    for i in range(1,f.size-1):
        d2[i] = (f[i+1] + f[i-1] - 2.0*f[i])/(dx*dx)
    d2[0] = (f[1]-2.0*f[0])/(dx*dx)
    d2[f.size-1] = (f[f.size-2]-2.0*f[f.size-1])/(dx*dx)
    return d2
 
# Define a figure to take two plots
fig = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
ax = fig.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2,2))
ax.set_title("Basis set before orthonormalisation")
ax2 = fig.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-2,2))
ax2.set_title("Basis set after orthonormalisation")

for i in range(num_basis):
    basis_array[i,:] = poly(x,i,width)
    norm = integrate_functions(basis_array[i,:],basis_array[i,:],num_x_points,dx)
    basis_array[i,:] = basis_array[i,:]/np.sqrt(norm)
    ax.plot(x,basis_array[i,:])
    if i>0:
        for j in range(i):
            overlap = integrate_functions(basis_array[i,:],basis_array[j,:],num_x_points,dx)
            basis_array[i,:] = basis_array[i,:] - overlap*basis_array[j,:]
    norm = integrate_functions(basis_array[i,:],basis_array[i,:],num_x_points,dx)
    basis_array[i,:] = basis_array[i,:]/np.sqrt(norm)
    d2basis_array[i,:] = d2_finite(basis_array[i,:],dx)
    ax2.plot(x,basis_array[i,:])
    
print "Now check that the basis set is orthonormal"
print
for i in range(num_basis):
    for j in range(num_basis):
        overlap = integrate_functions(basis_array[i,:],basis_array[j,:],num_x_points,dx)
        print "%8.3f" % overlap,
    print

# First let's solve the simple square well
# Declare space for the matrix elements - simplest with the identity function
Hmat = np.eye(num_basis)

# Define a function to act on a basis function with the potential
def add_pot_on_basis(Hphi,V,phi):
    for i in range(V.size):
        Hphi[i] = Hphi[i] + V[i]*phi[i]
        
print "Full Hamiltonian"
# Loop over basis functions phi_n (the bra in the matrix element)
# Calculate and store the matrix elements for the full Hamiltonian
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # Act with H on phi_m and store in H_phi_m
        # First the kinetic energy
        H_phi_m = -0.5*d2basis_array[m] 
        # Potential is zero for the pure square well
        # Create matrix element by integrating
        H_mn = integrate_functions(basis_array[n],H_phi_m,num_x_points,dx)
        Hmat[m,n] = H_mn
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % H_mn,
    # This print puts in a new line when we have finished looping over m
    print

# Solve using linalg module of numpy (which we've imported as la above)
eigval, eigvec = la.eigh(Hmat)
# This call above does the entire solution for the eigenvalues and eigenvectors !
# Print results roughly, though apply precision of 4 to the printing
np.set_printoptions(precision=4)
print eigval
print eigvec[0]
print eigvec[1]
print eigvec[2]

# Now print out eigenvalues and the eigenvalues of the perfect square well, and the difference
print " Changed Original  Difference"
for i in range(num_basis):
    n = i+1
    print "%8.3f %8.3f %8.3f" % (eigval[i],n*n*np.pi*np.pi/2.0,eigval[i] - n*n*np.pi*np.pi/2.0)

# Define a figure to take two plots
fig = pl.figure(figsize=[12,3])
# Add subplots: number in y, x, index number
ax = fig.add_subplot(121,autoscale_on=False,xlim=(0,1),ylim=(-2,2))
ax.set_title("Eigenvectors for changed system")
ax2 = fig.add_subplot(122,autoscale_on=False,xlim=(0,1),ylim=(-0.002,0.002))
ax2.set_title("Difference to perfect eigenvectors")
for m in range(4): # Plot the first four states
    psi = np.zeros(num_x_points)
    for i in range(num_basis):
        psi = psi+eigvec[i,m]*basis_array[i]
    if psi[1] < 0:  # This is just to ensure that psi and the basis function have the same phase
        psi = -psi
    ax.plot(x,psi)
    fac = np.pi*(m+1)/width
    exact = np.sin(fac*x)
    norm = integrate_functions(exact,exact,num_x_points,dx)
    exact = exact/np.sqrt(norm)
    psi = psi - exact
    ax2.plot(x,psi)

# Define the eigenbasis - normalisation needed elsewhere
def eigenbasis_sw(n,width,norm,x):
    """The eigenbasis for a square well, running from 0 to a, sin(n pi x/a)"""
    fac = np.pi*n/width
    return norm*np.sin(fac*x)

# These arrays will each hold an array of functions
alt_basis_array = np.zeros((num_basis,num_x_points))

# Loop over first num_basis basis states, normalise and create an array
# NB the basis_array will start from 0
for i in range(num_basis):
    n = i+1
    # Calculate A = <phi_n|phi_n>
    integral = integrate_functions(eigenbasis_sw(n,width,1.0,x),eigenbasis_sw(n,width,1.0,x),num_x_points,dx)
    # Use 1/sqrt{A} as normalisation constant
    normalisation = 1.0/np.sqrt(integral)
    alt_basis_array[i,:] = eigenbasis_sw(n,width,normalisation,x)

# Now create the similarity matrix, S
Smat = np.eye(num_basis)
# Loop over basis functions chi_a (the bra in the matrix element)
# Calculate and store the matrix elements 
for a in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        Smat[a,m] = integrate_functions(alt_basis_array[a],basis_array[m],num_x_points,dx)
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % Smat[a,m],
    # This print puts in a new line when we have finished looping over m
    print

print "Calculating S.H then S.H.ST: "
SH = np.dot(Smat,Hmat)
alt_H = np.dot(SH,Smat.T)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % alt_H[n,m],
    # This print puts in a new line when we have finished looping over m
    print
    
# There is no need to do this - it should give exactly the same result
# But it is a useful consistency check and may give a hint to any asymmetries
print "Calculating H.ST then S.H.ST: "
HS = np.dot(Hmat,Smat.T)
alt_H2 = np.dot(Smat,HS)
for n in range(num_basis):
    # Loop over basis functions phi_m (the ket in the matrix element)
    for m in range(num_basis):
        # The comma at the end prints without a new line; the %8.3f formats the number
        print "%8.3f" % alt_H2[n,m],
    # This print puts in a new line when we have finished looping over m
    print

# Print the diagonal of the Hamiltonian from polynomial basis (Poly), the exact eigenvalue (Eigen) and the difference
print "    Poly    Eigen   Difference"
for i in range(num_basis):
    n = i+1
    print "%8.3f %8.3f %8.3f" %(alt_H[i,i], n*n*np.pi*np.pi/2.0,alt_H[i,i]-n*n*np.pi*np.pi/2.0)



