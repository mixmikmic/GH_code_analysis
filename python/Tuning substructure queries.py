from rdkit import Chem,DataStructs
import time,random
from collections import defaultdict
import psycopg2
from rdkit.Chem import Draw,PandasTools,rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw,rdqueries
from rdkit import rdBase
from __future__ import print_function
import requests
from xml.etree import ElementTree
import pandas as pd
get_ipython().magic('load_ext sql')
print(rdBase.rdkitVersion)
print(time.asctime())

data = get_ipython().magic("sql postgresql://localhost/chembl_20     select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>'c1ccccc1*'::qmol limit 10;")

mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)

sma = '[cH]1[cH][cH][cH][cH]c1*'
data = get_ipython().magic('sql      select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>:sma ::qmol limit 10;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)            

sma = 'c1ccccc1-[*]'
m = Chem.MolFromSmarts(sma)
# start by creating an RWMol:
m = Chem.RWMol(m)
for atom in m.GetAtoms():
    # skip dummies:
    if not atom.GetAtomicNum():
        continue
    atom.ExpandQuery(rdqueries.ExplicitDegreeEqualsQueryAtom(atom.GetDegree()))
sma = Chem.MolToSmarts(m)
print("Result SMARTS:",sma)

data = get_ipython().magic('sql      select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>:sma ::qmol limit 10;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)            

sma = 'c1ccccc1-[*]'
m = Chem.MolFromSmiles(sma)
# start by creating an RWMol:
m = Chem.RWMol(m)
for atom in m.GetAtoms():
    # skip dummies:
    if not atom.GetAtomicNum():
        continue
    oa = atom
    if not atom.HasQuery():
        needsReplacement=True
        atom = rdqueries.AtomNumEqualsQueryAtom(oa.GetAtomicNum())
        atom.ExpandQuery(rdqueries.IsAromaticQueryAtom(oa.GetIsAromatic()))
        if(oa.GetIsotope()):
            atom.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(oa.GetIsotope()))
        if(oa.GetFormalCharge()):
            atom.ExpandQuery(rdqueries.FormalChargeEqualsQueryAtom(oa.GetFormalCharge()))
    else:
        needsReplacement=False
    atom.ExpandQuery(rdqueries.ExplicitDegreeEqualsQueryAtom(oa.GetDegree()))
    if needsReplacement:
        m.ReplaceAtom(oa.GetIdx(),atom)
sma = Chem.MolToSmarts(m)
print("Result SMARTS:",sma)

data = get_ipython().magic('sql      select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>:sma ::qmol limit 10;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)

def adjustQuery(m,ringsOnly=True,ignoreDummies=True):
    qm =Chem.RWMol(m)
    if ringsOnly:           
        ri = qm.GetRingInfo()
        try:
            ri.NumRings()
        except RuntimeError:
            ri=None
            Chem.FastFindRings(qm)
            ri = qm.GetRingInfo()
    for atom in qm.GetAtoms():
        if ignoreDummies and not atom.GetAtomicNum():
            continue
        if ringsOnly and not ri.NumAtomRings(atom.GetIdx()):
            continue

        oa = atom
        if not atom.HasQuery():
            needsReplacement=True
            atom = rdqueries.AtomNumEqualsQueryAtom(oa.GetAtomicNum())
            atom.ExpandQuery(rdqueries.IsAromaticQueryAtom(oa.GetIsAromatic()))
            if(oa.GetIsotope()):
                atom.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(oa.GetIsotope()))
            if(oa.GetFormalCharge()):
                atom.ExpandQuery(rdqueries.FormalChargeEqualsQueryAtom(oa.GetFormalCharge()))
        else:
            needsReplacement=False
        atom.ExpandQuery(rdqueries.ExplicitDegreeEqualsQueryAtom(oa.GetDegree()))
        if needsReplacement:
            qm.ReplaceAtom(oa.GetIdx(),atom)            
    return qm

qm = adjustQuery(Chem.MolFromSmarts('c1ccccc1*'))
sma = Chem.MolToSmarts(qm)
print("Result SMARTS:",sma)
data = get_ipython().magic('sql      select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>:sma ::qmol limit 10;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)

sma = 'Oc1cc(*)ccc1'
qm = adjustQuery(Chem.MolFromSmarts(sma))
sma = Chem.MolToSmarts(qm)
print("Result SMARTS:",sma)
data = get_ipython().magic('sql      select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>:sma ::qmol limit 10;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)

sma = 'Oc1cc(*)ccc1'
qm = adjustQuery(Chem.MolFromSmarts(sma),ringsOnly=False)
sma = Chem.MolToSmarts(qm)
print("Result SMARTS:",sma)
data = get_ipython().magic('sql      select molregno,molfile from rdk.mols join compound_structures using (molregno) where m@>:sma ::qmol limit 10;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)



