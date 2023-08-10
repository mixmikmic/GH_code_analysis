from IPython.display import Image
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import json
import collections
from pymatgen import Structure
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer,     get_arrhenius_plot, get_extrapolated_conductivity
from pymatgen_diffusion.aimd.pathway import ProbabilityDensityAnalysis
from pymatgen_diffusion.aimd.van_hove import VanHoveAnalysis

# files = ["run1/vasprun.xml", "run2/vasprun.xml", "run3/vasprun.xml"]
# analyzer = DiffusionAnalyzer.from_files(files, specie="Li", smoothed=False)

temperatures = [600, 800, 1000, 1200]
analyzers = collections.OrderedDict()
for temp in temperatures:
    with open("aimd_data/%d.json" % temp) as f:
        d = json.load(f)
        analyzers[temp] = DiffusionAnalyzer.from_dict(d)

plt = analyzers[1000].get_msd_plot()
title = plt.title("1000K", fontsize=24)

diffusivities = [d.diffusivity for d in analyzers.values()]

plt = get_arrhenius_plot(temperatures, diffusivities)

rts = get_extrapolated_conductivity(temperatures, diffusivities, 
                                    new_temp=300, structure=analyzers[800].structure, 
                                    species="Li")
print("The Li ionic conductivity for Li6PS5Cl at 300 K is %.4f mS/cm" % rts)

structure = analyzers[800].structure
trajectories = [s.frac_coords for s in analyzers[800].get_drift_corrected_structures()]
pda = ProbabilityDensityAnalysis(structure, trajectories, species="Li")
pda.to_chgcar("aimd_data/CHGCAR.vasp") # Output to a CHGCAR-like file for visualization in VESTA.

Image(filename='Isosurface_800K_0.png') 

vha = VanHoveAnalysis(analyzers[800])  

vha.get_3d_plot(type="self")
vha.get_3d_plot(type="distinct")



