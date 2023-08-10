import rdkit # compchem library
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole # use this to draw molecules in the notebook
from rdkit import rdBase
print(rdBase.rdkitVersion)

import py3Dmol # for 3D viz

Chem.MolFromSmiles('[H]O[H]') # a simple molecule, water

# a little more complex example - coffee (caffeine) -
Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

m = Chem.AddHs(Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C'))
AllChem.EmbedMultipleConfs(m ,useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
mb = Chem.MolToMolBlock(m)
p = py3Dmol.view(width=400,height=400)
p.addModel(mb, 'sdf')
p.setStyle({'stick':{}})
p.setBackgroundColor('0xffffff')
p.zoomTo()
p.show()

# Proteins are much larger complexes, sometimes called macromolecules
view = py3Dmol.view(query='pdb:1pwc', options={'multimodel':True})
view.setStyle({'chain':'A'}, {'stick':{'color':'spectrum'}})
#view.addSurface("VDW", {'opacity': 0.6});
view

view = py3Dmol.view(query='pdb:1pwc')
view.setStyle({'chain':'A'}, {'stick':{}})
view.setStyle({'resn':'PNM'}, {'stick':{'colorscheme':'yellowCarbon'}})
view.addSurface("VDW", {'opacity': 0.9},{'hetflag':False});
view.zoomTo({'resn':'PNM'})
view.show()

