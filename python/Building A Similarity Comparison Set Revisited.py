from rdkit import Chem
from rdkit import rdBase
print(rdBase.rdkitVersion)
import time
print(time.asctime())

import psycopg2
cn = psycopg2.connect(dbname='chembl_21')
curs = cn.cursor()
curs.execute('select molregno,m from rdk.mols join rdk.tfps_smaller using (molregno) order by random() limit 35000')
qs = curs.fetchall()

cn.rollback()
curs.execute('set rdkit.tanimoto_threshold=0.7')

keep=[]
for i,row in enumerate(qs):
    curs.execute('select molregno,m from rdk.mols join (select molregno from rdk.tfps_smaller where mfp0%%morgan_fp(%s,0) '
                 'and molregno!=%s limit 1) t2 using (molregno)',(row[1],row[0]))
    d = curs.fetchone()
    if not d: continue
    keep.append((row[0],row[1],d[0],d[1]))
    if len(keep)==25000: break
    if not i%1000: print('Done: %d'%i)

import gzip
outf = gzip.open('../data/chembl21_25K.pairs.txt.gz','wb+')
for idx1,smi1,idx2,smi2 in keep: outf.write(('%d %s %d %s\n'%(idx1,smi1,idx2,smi2)).encode('UTF-8'))
outf=None

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True
from rdkit.Chem import Draw
import gzip

rows=[]
for row in gzip.open('../data/chembl21_25K.pairs.txt.gz').readlines():
    row = row.split()
    row[1] = Chem.MolFromSmiles(row[1])
    row[3] = Chem.MolFromSmiles(row[3])
    rows.append(row)

t = []
for x in rows[:5]:
    t.append(x[1])
    t.append(x[3])
    
Draw.MolsToGridImage(t,molsPerRow=2)

from rdkit.Chem import Descriptors

mws = [(Descriptors.MolWt(x[1]),Descriptors.MolWt(x[3])) for x in rows]
nrots = [(Descriptors.NumRotatableBonds(x[1]),Descriptors.NumRotatableBonds(x[3])) for x in rows]
logps = [(Descriptors.MolLogP(x[1]),Descriptors.MolLogP(x[3])) for x in rows]

get_ipython().magic('pylab inline')

_=hist(([x for x,y in mws],[y for x,y in mws]),bins=20,histtype='bar')
xlabel('AMW')

_=hist(([x for x,y in logps],[y for x,y in logps]),bins=20,histtype='bar')
xlabel('mollogp')

_=hist(([x for x,y in nrots],[y for x,y in nrots]),bins=20,histtype='bar')
xlabel('num rotatable bonds')

from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
sims = [DataStructs.TanimotoSimilarity(rdMolDescriptors.GetMorganFingerprint(x[1],0),rdMolDescriptors.GetMorganFingerprint(x[3],0)) for x in rows]

_=hist(sims,bins=20)
xlabel('MFP0 sims within pairs')

sims2 = [DataStructs.TanimotoSimilarity(rdMolDescriptors.GetMorganFingerprint(x[1],2),rdMolDescriptors.GetMorganFingerprint(x[3],2)) for x in rows]

_=scatter(sims,sims2,marker='o',edgecolors='none')
xlabel('MFP0 sim')
ylabel('MFP2 sim')

import random
idxs = list(range(len(rows)))
random.shuffle(idxs)
ms1 = [x[1] for x in rows]
ms2 = [rows[x][3] for x in idxs]
sims = [DataStructs.TanimotoSimilarity(rdMolDescriptors.GetMorganFingerprint(x,0),rdMolDescriptors.GetMorganFingerprint(y,0)) for x,y in zip(ms1,ms2)]

_=hist(sims,bins=20)
xlabel('MFP0 sim in random pairs')

cn = None
curs=None

import psycopg2
cn = psycopg2.connect(dbname='chembl_21')
curs = cn.cursor()
curs.execute('select molregno,m from rdk.mols join rdk.tfps1_smaller using (molregno) order by random() limit 35000')
qs = curs.fetchall()

cn.rollback()
curs.execute('set rdkit.tanimoto_threshold=0.6')

keep=[]
for i,row in enumerate(qs):
    curs.execute('select molregno,m from rdk.mols join (select molregno from rdk.tfps1_smaller where mfp1%%morgan_fp(%s,1) '
                 'and molregno!=%s limit 1) t2 using (molregno)',(row[1],row[0]))
    d = curs.fetchone()
    if not d: continue
    keep.append((row[0],row[1],d[0],d[1]))
    if len(keep)==25000: break
    if not i%1000: print('Done: %d'%i)

import gzip
outf = gzip.open('../data/chembl21_25K.mfp1.pairs.txt.gz','wb+')
for idx1,smi1,idx2,smi2 in keep: outf.write(('%d %s %d %s\n'%(idx1,smi1,idx2,smi2)).encode('UTF-8'))
outf=None

print(len(keep))

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True
from rdkit.Chem import Draw
import gzip

rows=[]
for row in gzip.open('../data/chembl21_25K.mfp1.pairs.txt.gz').readlines():
    row = row.split()
    row[1] = Chem.MolFromSmiles(row[1])
    row[3] = Chem.MolFromSmiles(row[3])
    rows.append(row)
    if len(rows)>100: break # we aren't going to use all the pairs, so there's no sense in reading them all in

t = []
for x in rows[:5]:
    t.append(x[1])
    t.append(x[3])
    
Draw.MolsToGridImage(t,molsPerRow=2)

