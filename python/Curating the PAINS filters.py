from rdkit import Chem
import time,random
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.molSize = (450,250)
from rdkit import rdBase
from __future__ import print_function
get_ipython().magic('load_ext sql')
print(rdBase.rdkitVersion)
print(time.asctime())

sma = '[#8]=[#16](=[#8])-[#6](-[#6]#[#7])=[#7]-[#7]-[#1]'
pains8 = Chem.MolFromSmarts(sma)
pains8

mol8 = Chem.MolFromSmiles(r'COC(=O)c1sc(SC)c(S(=O)(=O)C(C)C)c1N/N=C(\C#N)S(=O)(=O)c1ccccn1') #CHEMBL3211428
mol8

mol8.HasSubstructMatch(pains8)

mol8h = Chem.AddHs(mol8)
mol8h.HasSubstructMatch(pains8)

pains8h = Chem.MergeQueryHs(pains8)
mol8.HasSubstructMatch(pains8h)

mol8h.HasSubstructMatch(pains8h)

Chem.MolToSmarts(pains8h)

patt = Chem.MolFromSmarts('[#6]([#1])[#1]',mergeHs=True)
Chem.MolToSmarts(patt)

patt = Chem.MolFromSmarts('[$([#6]-[#7]),$([#6]-[#1]),$([#6])]',mergeHs=True)
Chem.MolToSmarts(patt)

patt = Chem.MolFromSmarts('[#6]-[#1,#6]',mergeHs=True)
Chem.MolToSmarts(patt)



