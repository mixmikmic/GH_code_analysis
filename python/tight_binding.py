from pytriqs.lattice.tight_binding import *

BL = BravaisLattice(units = [(1,0,0), (0,1,0)]) 

t   = -1.00                # First neighbour Hopping
tp  =  0.0*t               # Second neighbour Hopping

hop= {  (1,0)  :  [[ t]],       
        (-1,0) :  [[ t]],     
        (0,1)  :  [[ t]],
        (0,-1) :  [[ t]],
        (1,1)  :  [[ tp]],
        (-1,-1):  [[ tp]],
        (1,-1) :  [[ tp]],
        (-1,1) :  [[ tp]]}

TB = TightBinding(BL, hop)

density_states = dos(TB, n_kpts= 500, n_eps = 101, name = 'dos')[0]

from pytriqs.plot.mpl_interface import oplot,plt
get_ipython().magic('matplotlib inline')
oplot(density_states, '-o')
plt.xlim (-5,5)
plt.ylim (0, 0.4)



