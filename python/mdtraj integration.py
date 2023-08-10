import mdtraj as md
traj = md.load_pdb('../tests/data/2M6K.pdb')
print(traj)

from chemview import MolecularViewer
from chemview.contrib import topology_mdtraj

mv = MolecularViewer(traj.xyz[0], topology_mdtraj(traj))
# mv.cylinder_and_strand()
mv.cartoon()
mv

from chemview.export import display_static

display_static(mv)

