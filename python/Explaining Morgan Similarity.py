import numpy
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import rdBase
print(rdBase.rdkitVersion)

import time
print(time.asctime())
get_ipython().magic('pylab inline')

#
# Functions for providing detailed descriptions of MFP bits 
#  inspired by some code from from Nadine Schneider 
#
def getSubstructSmi(mol,atomID,radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        symbol = '['+atom.GetSmarts()
        if nHs: 
            symbol += 'H'
            if nHs>1:
                symbol += '%d'%nHs
        if isInRing:
            symbol += ';R'
        else:
            symbol += ';!R'
        symbol += ';D%d'%deg
        symbol += "]"
        symbols.append(symbol)
    smi = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
    smi2 = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)
    return smi,smi2

# Start by importing some code to allow the depiction to be used:
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D

# a function to make it a bit easier. This should probably move to somewhere in
# rdkit.Chem.Draw
def _prepareMol(mol,kekulize):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc
def moltosvg(mol,molSize=(450,200),kekulize=True,drawer=None,**kwargs):
    mc = rdMolDraw2D.PrepareMolForDrawing(mol,kekulize=kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return SVG(svg.replace('svg:',''))

# do a depiction where the atom environment is highlighted normally and the central atom
# is highlighted in blue
def getSubstructDepiction(mol,atomID,radius,molSize=(450,200)):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))       
    else:
        atomsToUse = [atomID]
        env=None
    return moltosvg(mol,molSize=molSize,highlightAtoms=atomsToUse,highlightAtomColors={atomID:(0.3,0.3,1)})
def depictBit(bitId,mol,molSize=(450,200)):
    info={}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048,bitInfo=info)
    aid,rad = info[bitId][0]
    return getSubstructDepiction(mol,aid,rad,molSize=molSize)

bz = Chem.MolFromSmiles('c1ccccc1')
fp_bz = AllChem.GetMorganFingerprintAsBitVect(bz,radius=2,nBits=2048)
pyr = Chem.MolFromSmiles('c1ccccn1')
fp_pyr = AllChem.GetMorganFingerprintAsBitVect(pyr,radius=2,nBits=2048)
print("Similarity:",DataStructs.TanimotoSimilarity(fp_bz,fp_pyr))

print("intersection count:",(fp_bz&fp_pyr).GetNumOnBits())
print("union count:",(fp_bz|fp_pyr).GetNumOnBits())

bi_bz = {}
fp_bz = AllChem.GetMorganFingerprintAsBitVect(bz,radius=2,nBits=2048,bitInfo=bi_bz)
bi_pyr = {}
fp_pyr = AllChem.GetMorganFingerprintAsBitVect(pyr,radius=2,nBits=2048,bitInfo=bi_pyr)

bi_bz

info_bz = []
for bitId,atoms in bi_bz.items():
    exampleAtom,exampleRadius = atoms[0]
    description = getSubstructSmi(bz,exampleAtom,exampleRadius)
    info_bz.append((bitId,exampleRadius,description[0],description[1]))
print(info_bz)

info_pyr = []
for bitId,atoms in bi_pyr.items():
    exampleAtom,exampleRadius = atoms[0]
    description = getSubstructSmi(pyr,exampleAtom,exampleRadius)
    info_pyr.append((bitId,exampleRadius,description[0],description[1]))
print(info_pyr)

collection = {}
for bid,rad,smi,sma in info_bz:
    collection[bid] = [bid,rad,smi,sma,'','']
for bid,rad,smi,sma in info_pyr:
    if bid not in collection:
        collection[bid] = [bid,rad,'','','','']
    collection[bid][-2] = smi
    collection[bid][-1] = sma

import pandas as pd
pd.options.display.width=100000 # options to make sure our wide columns display properly
pd.options.display.max_colwidth=1000

df = pd.DataFrame(list(collection.values()),columns=('Bit','radius','smi_bz','sma_bz','smi_pyr','sma_pyr'))
df.sort_values(by='radius')

depictBit(1866,pyr,molSize=(125,125))

depictBit(1155,pyr,molSize=(125,125))

depictBit(383,pyr,molSize=(125,125))

