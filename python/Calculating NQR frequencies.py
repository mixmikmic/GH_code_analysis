from __future__ import print_function

from magres.atoms import MagresAtoms
from magres.constants  import millibarn, megahertz
import numpy, math

def available_m(spin):
     return [m for m in numpy.arange(-spin, spin+1, 1) if m >= 0.0][:-1]

print ("S=0.5, available |m| = ", available_m(0.5))
print ("S=1.0, available |m| = ", available_m(1.0))
print ("S=1.5, available |m| = ", available_m(1.5))
print ("S=2.0, available |m| = ", available_m(2.0))

def calc_nqr(atom, m):
    efg = atom.efg
    Q = atom.Q
    spin = atom.spin
    
    Vzz = efg.evals[2]
    vec_zz = efg.evecs[2]
    eta = (abs(efg.evals[0]) - abs(efg.evals[1]))/efg.evals[2]
    A = Vzz * (Q * millibarn) / (4.0 * spin * (2.0*spin - 1.0))
    fq = 3*A * (2.0*abs(m) + 1.0) * math.sqrt(1.0 + eta**2/3)

    return fq

atoms = MagresAtoms.load_magres('../samples/NaClO3.magres')

freqs = []

for Cl_atom in atoms.species('Cl'):
    print(Cl_atom, "S={:.1f} Q={:.2f} millibarn".format(Cl_atom.spin, Cl_atom.Q))
    
    for m in available_m(Cl_atom.spin):
        freq_MHz = calc_nqr(Cl_atom, m) / megahertz
        print("  m={:.1f}->{:.1f} freq = {:.3f} MHz".format(m, m+1, freq_MHz))
        
        freqs.append(freq_MHz)
        
print("Mean freq = {:.3f} Mhz".format(numpy.mean(freqs)))

