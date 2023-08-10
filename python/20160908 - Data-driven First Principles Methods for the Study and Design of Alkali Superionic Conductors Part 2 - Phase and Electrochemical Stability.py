get_ipython().magic('matplotlib inline')

from pymatgen import MPRester, Composition, Element
from pymatgen.io.vasp import Vasprun
from pymatgen.phasediagram.maker import PhaseDiagram, CompoundPhaseDiagram
from pymatgen.phasediagram.analyzer import PDAnalyzer
from pymatgen.phasediagram.plotter import PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.util.plotting_utils import get_publication_quality_plot
import json
import re
import palettable
import matplotlib as mpl

vasprun = Vasprun("aimd_data/vasprun.xml.relax2.gz")
# include structure so proper correction can be applied for oxides and sulfides
entry = vasprun.get_computed_entry(inc_structure=True)  

rester = MPRester()
mp_entries = rester.get_entries_in_chemsys(["Li", "P", "S", "Cl"])

with open("aimd_data/lpo_entries.json") as f:
    lpo_data = json.load(f)
lpo_entries = [ComputedEntry.from_dict(d) for d in lpo_data]

compatibility = MaterialsProjectCompatibility()
entry = compatibility.process_entry(entry)
entries = compatibility.process_entries([entry] + mp_entries + lpo_entries)

pd = PhaseDiagram(entries)
plotter = PDPlotter(pd)
plotter.show()

cpd = CompoundPhaseDiagram(entries, 
                           [Composition("P2S5"), Composition("Li2S"), Composition("LiCl")])
cplotter = PDPlotter(cpd, show_unstable=True)
cplotter.show()

analyzer = PDAnalyzer(pd)
ehull = analyzer.get_e_above_hull(entry)
print("The energy above hull of Li6PS5Cl is %.3f eV/atom." % ehull)

li_entries = [e for e in entries if e.composition.reduced_formula == "Li"]
uli0 = min(li_entries, key=lambda e: e.energy_per_atom).energy_per_atom

el_profile = analyzer.get_element_profile(Element("Li"), entry.composition)
for i, d in enumerate(el_profile):
    voltage = -(d["chempot"] - uli0)
    print("Voltage: %s V" % voltage)
    print(d["reaction"])
    print("")

# Some matplotlib settings to improve the look of the plot.
mpl.rcParams['axes.linewidth']=3
mpl.rcParams['lines.markeredgewidth']=4
mpl.rcParams['lines.linewidth']=3
mpl.rcParams['lines.markersize']=15
mpl.rcParams['xtick.major.width']=3
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['xtick.minor.width']=3
mpl.rcParams['xtick.minor.size']=4
mpl.rcParams['ytick.major.width']=3
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['ytick.minor.width']=3
mpl.rcParams['ytick.minor.size']=4


# Plot of Li uptake per formula unit (f.u.) of Li6PS5Cl against voltage vs Li/Li+.

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
plt = get_publication_quality_plot(12, 8)

for i, d in enumerate(el_profile):
    v = - (d["chempot"] - uli0)
    if i != 0:
        plt.plot([x2, x2], [y1, d["evolution"] / 4.0], 'k', linewidth=3)
    x1 = v
    y1 = d["evolution"] / 4.0
    if i != len(el_profile) - 1:
        x2 = - (el_profile[i + 1]["chempot"] - uli0)
    else:
        x2 = 5.0
        
    if i in [0, 4, 5, 7]:
        products = [re.sub(r"(\d+)", r"$_{\1}$", p.reduced_formula)                     
                    for p in d["reaction"].products if p.reduced_formula != "Li"]

        plt.annotate(", ".join(products), xy=(v + 0.05, y1 + 0.3), 
                     fontsize=24, color=colors[0])
        
        plt.plot([x1, x2], [y1, y1], color=colors[0], linewidth=5)
    else:
        plt.plot([x1, x2], [y1, y1], 'k', linewidth=3)  

plt.xlim((0, 4.0))
plt.ylim((-6, 10))
plt.xlabel("Voltage vs Li/Li$^+$ (V)")
plt.ylabel("Li uptake per f.u.")
plt.tight_layout()



