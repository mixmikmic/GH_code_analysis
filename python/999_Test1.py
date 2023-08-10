from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True

import time
print(time.asctime()) # doctest: IGNORE

Chem.CanonSmiles("c1ccccc1O")

Chem.MolFromSmiles('c1ccccc1O')

print("hello")
# comment
Chem.CanonSmiles("c1ccccc1NC")

frags = Chem.FragmentOnBonds(Chem.MolFromSmiles('CCOC1CC1'),(2,))
frags

Chem.MolToSmiles(frags,True)

pieces = Chem.GetMolFrags(frags,asMols=True)
len(pieces)

[Chem.MolToSmiles(x,True) for x in pieces]

from rdkit.Chem import AllChem
m = Chem.MolFromSmiles('CCOC1CC1')
m.SetProp("_Name","test molecule")
AllChem.Compute2DCoords(m)

print(Chem.MolToMolBlock(m))

# test that for loops work:
atns = []
for at in m.GetAtoms():
    if at.GetAtomicNum()>6:
        atns.append((at.GetIdx(),at.GetAtomicNum()))
atns

atns = []
for at in m.GetAtoms():
    if at.GetAtomicNum()>6:
        atns.append((at.GetIdx(),at.GetAtomicNum()))
atns

for at in m.GetAtoms():
    if at.GetAtomicNum()>6:
        atns.append((at.GetIdx(),at.GetAtomicNum()))
if(len(atns)==1):        
    atns



