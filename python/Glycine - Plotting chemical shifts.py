from __future__ import print_function
import warnings
warnings.filterwarnings("ignore") 
get_ipython().magic('pylab inline')

from IPython.core.display import Image
Image(filename='glycine-relaxed.png', width=600, embed=True)

from magres.atoms import MagresAtoms, MagresAtomsView

atoms = MagresAtoms.load_magres('../samples/glycine-relaxed.magres')
atoms.calculate_bonds(1.6)

print("We have {} atoms".format(len(atoms)))

atoms.within(atoms.C2, 3.0)

atoms.species('H').set_reference(30)
H_cs = atoms.species('H').ms.cs
print("H min: {:.3f} max: {:.3f}".format(min(H_cs), max(H_cs)))

import numpy, scipy, scipy.stats
import matplotlib.pyplot as plt

H_xs = numpy.arange(0, 15, 0.01)
spread = 0.05
H_ys = numpy.sum([scipy.stats.norm.pdf(H_xs, cs, spread) for cs in H_cs], axis=0)

plt.gca().invert_xaxis()
plot(H_xs, H_ys)
show()

# Mean shifts on the H bonded to the N in NH3
H_cs1 = mean(atoms.N1.bonded.species('H').ms.cs)

# Just take the shifts on the H bonded to the C in CH2 as they are
H_cs2, H_cs3 = atoms.C2.bonded.species('H').ms.cs

# Make a list of them with weights
H_cs_ave = [(3,H_cs1), (1,H_cs2), (1,H_cs3)]

print("Average 1H shifts:", H_cs_ave)

H_ys_ave = numpy.sum([strength*scipy.stats.norm.pdf(H_xs, cs, spread) for strength,cs in H_cs_ave], axis=0)

plt.gca().invert_xaxis()
plot(H_xs, H_ys_ave)
show()

print("CH2 separation = {:.3f}, exp ~ 1ppm".format(abs(H_cs2 - H_cs3)))
print("NH3-CH2 separation = {:.3f}, exp ~ 5ppm".format(abs(H_cs1 - (H_cs2 + H_cs3)/2)))

