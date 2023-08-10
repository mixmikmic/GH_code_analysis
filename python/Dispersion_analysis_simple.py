from __future__ import division#, print_function
import numpy as np
from scipy.constants import c,pi
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
import matplotlib.pylab as plt
from matplotlib.colors import from_levels_and_colors
import time
from functions_dispersion_analysis import *

get_ipython().magic('matplotlib inline')

a = 1e-4
b = 1e-4

mu_r = 1.0
lamda = 1.55e-6
r_core = 0.2e-5 # radius of core
r_clad = 5e-5 #radius of the fibre
nclad = 1.444# - 0.1e-4j# ref index of cladding
ncore = 1.5# - 1e-4j # ref index of core
neff_g = ncore # Guess of the modes
num= 20   #The number of modes guess 
#neff_g= ncore
lam_mult = 1
# Asks GMSH to create a mesh that has this number multiplied by the wavelength 
mesh_refinement = 0 # number of times to uniformly refine the mesh (used for convergence plots and better results)
vector_order = 3
nodal_order = 3

#from testing.Single_mode_fibre.Single_mode_theoretical import *
#neff_th, Aeff_th = main_test(ncore,nclad,lamda,r_core,r_clad)
k =0 

if k ==0:
    V = 2*pi/lamda*r_core*(ncore**2 - nclad**2)**0.5
    print(V)

def ref(x,values = np.zeros(1)):
    point = (x[0]**2+ x[1]**2)**0.5
    if  point<= r_core:
        values[0] = ncore.real**2 - ncore.imag**2
    elif point > r_core and point <= r_clad:
        values[0] = nclad.real**2 - nclad.imag**2
    else:
        values[0] = 1.
    return values

def extinction(x,values = np.zeros(1)):
    point = (x[0]**2+ x[1]**2)**0.5
    if  point<= r_core:
        values[0] = -2*ncore.imag*ncore.real
    elif point > r_core and point <= r_clad:
        values[0] = -2*nclad.imag*ncore.real
    else:
        values[0] = 0
    return values

waveguide = waveguide_inputs(lamda,ref,extinction)
waveguide.fibre(True,r_core,r_clad,ncore,nclad)
k = is_loss(ncore,nclad)
box_domain = box_domains(a,a)

mesh = None

#min_max = (-box_domain.a, box_domain.a,-box_domain.b, box_domain.b)


modes_vec = main(box_domain,waveguide,vector_order,nodal_order,num,neff_g,lam_mult,min_max=None,k = 0,                 size1 = 512,size2 = 512, mesh_plot = False,filename = 'geometry_circular.geo')
mesh = modes_vec[-1]
modes_vec =modes_vec[:-1]

for mode in modes_vec:
    mode.plot_electric_field(scales = 150000*10,sp=30,cont_scale=700)
    #np.abs(neff_th - mode.neff).real
    #np.abs(neff_th - mode.neff).imag



