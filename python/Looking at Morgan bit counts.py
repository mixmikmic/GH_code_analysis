import numpy
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import pickle
from collections import Counter
from rdkit import rdBase
print(rdBase.rdkitVersion)

import time
print(time.asctime())
get_ipython().magic('pylab inline')

reader = DataStructs.FPBReader('/scratch/RDKit_git/Data/Zinc/zinc_all_clean.mfp2.fpb',lazy=True)
reader.Init()
print(len(reader))

numBitCount = Counter()
fpBitCount = Counter()
for i in range(len(reader)):
    fp = reader.GetFP(i)
    numBitCount[fp.GetNumOnBits()]+=1
    for bit in fp.GetOnBits():
        fpBitCount[bit]+=1
    if not i%100000:
        print("Doing: ",i)

plot([x for x,y in numBitCount.items()],[y for x,y in numBitCount.items()],'.b-')
_=xlabel("num bits set")
_=ylabel("count")

min(numBitCount.items()),max(numBitCount.items())

plot(sorted([y for x,y in fpBitCount.items()],reverse=True),'b-')
_=yscale("log")
_=ylabel("count")
_=title("count per bit, sorted")

[np.percentile(sorted([y for x,y in fpBitCount.items()]),z) for z in (1,25,50,75,99)]

min([y for x,y in fpBitCount.items()]),max([y for x,y in fpBitCount.items()])

keepMols={}
toFind = list(set(bitExamples.values()))
bitExamples={}
needed = list(range(reader.GetNumBits()))
suppl = Chem.SmilesMolSupplier('/tmp/zinc_all_clean.smi')
for i,m in enumerate(suppl):
    if not m:
        continue
    
    fp = Chem.GetMorganFingerprintAsBitVect(m,2,2048)
    mid = m.GetProp("_Name")
    for bit in fp.GetOnBits():
        if bit in needed:
            bitExamples[bit] = mid
            keepMols[mid]=m
            needed.remove(bit)
    if not len(needed):
        break
    if not i%10000:
        print("Done:",i," left:",len(needed))

pickle.dump((keepMols,bitExamples,numBitCount,fpBitCount),open("../data/mfp2_analysis.pkl",'wb+'))

(keepMols,bitExamples,numBitCount,fpBitCount) = pickle.load(open("../data/mfp2_analysis.pkl",'rb'))

itms = [(y,x) for x,y in fpBitCount.items()]

#
# Functions for providing detailed descriptions of MFP bits from Nadine Schneider 
#  It's probably better to do this using the atomSymbols argument but this does work.
#
def includeRingMembership(s, n):
    r=';R]'
    d="]"
    return r.join([d.join(s.split(d)[:n]),d.join(s.split(d)[n:])])
 
def includeDegree(s, n, d):
    r=';D'+str(d)+']'
    d="]"
    return r.join([d.join(s.split(d)[:n]),d.join(s.split(d)[n:])])
 
def writePropsToSmiles(mol,smi,order):
    #finalsmi = copy.deepcopy(smi)
    finalsmi = smi
    for i,a in enumerate(order):
        atom = mol.GetAtomWithIdx(a)
        if atom.IsInRing():
            finalsmi = includeRingMembership(finalsmi, i+1)
        finalsmi = includeDegree(finalsmi, i+1, atom.GetDegree())
    return finalsmi
 
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
    smi = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
    order = eval(mol.GetProp("_smilesAtomOutputOrder"))
    smi2 = writePropsToSmiles(mol,smi,order)
    return smi,smi2

for bCount,bitId in sorted(itms)[:20]:
    zid = bitExamples[bitId]
    if zid in keepMols:
        info={}
        fp = Chem.GetMorganFingerprintAsBitVect(keepMols[zid],2,2048,bitInfo=info)
        aid,rad = info[bitId][0]
        smi1,smi2 = getSubstructSmi(keepMols[zid],aid,rad)
        print(bitId,zid,rad,smi2,bCount)

for bCount,bitId in sorted(itms)[-20:]:
    zid = bitExamples[bitId]
    if zid in keepMols:
        info={}
        fp = Chem.GetMorganFingerprintAsBitVect(keepMols[zid],2,2048,bitInfo=info)
        aid,rad = info[bitId][0]
        smi1,smi2 = getSubstructSmi(keepMols[zid],aid,rad)
        print(bitId,zid,rad,smi2,bCount)

# Start by importing some code to allow the depiction to be used:
from IPython.display import SVG
from rdkit.Chem import rdDepictor
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
    mc = _prepareMol(mol,kekulize)
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
def depictBit(bitId,examples,mols,molSize=(450,200)):
    zid = examples[bitId]
    info={}
    fp = Chem.GetMorganFingerprintAsBitVect(mols[zid],2,2048,bitInfo=info)
    aid,rad = info[bitId][0]
    return getSubstructDepiction(mols[zid],aid,rad,molSize=molSize)

bCount,bitId = sorted(itms)[0]
depictBit(bitId,bitExamples,keepMols)

depictBit(sorted(itms)[1][1],bitExamples,keepMols)

depictBit(sorted(itms)[100][1],bitExamples,keepMols)

depictBit(sorted(itms)[-1][1],bitExamples,keepMols)

import pandas as pd
from rdkit.Chem import PandasTools
PandasTools.RenderImagesInAllDataFrames(images=True)

rows = []
for bCount,bitId in sorted(itms)[:20]:
    zid = bitExamples[bitId]
    if zid in keepMols:
        info={}
        fp = Chem.GetMorganFingerprintAsBitVect(keepMols[zid],2,2048,bitInfo=info)
        aid,rad = info[bitId][0]
        smi1,smi2 = getSubstructSmi(keepMols[zid],aid,rad)
        svg = depictBit(bitId,bitExamples,keepMols,molSize=(250,125))
        rows.append([bitId,zid,svg.data,bCount])

df = pd.DataFrame(rows,columns=('Bit','ZincID','drawing','count'))

df

rows = []
for bCount,bitId in sorted(itms,reverse=True)[:20]:
    zid = bitExamples[bitId]
    if zid in keepMols:
        info={}
        fp = Chem.GetMorganFingerprintAsBitVect(keepMols[zid],2,2048,bitInfo=info)
        aid,rad = info[bitId][0]
        smi1,smi2 = getSubstructSmi(keepMols[zid],aid,rad)
        svg = depictBit(bitId,bitExamples,keepMols,molSize=(250,125))
        rows.append([bitId,zid,svg.data,bCount])

pd.DataFrame(rows,columns=('Bit','ZincID','drawing','count'))



