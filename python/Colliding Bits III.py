from rdkit import Chem,DataStructs
import time,random,gzip,pickle,copy
import numpy as np
from collections import defaultdict
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.Draw import IPythonConsole
from rdkit import DataStructs
from rdkit import rdBase
get_ipython().magic('pylab inline')

print(rdBase.rdkitVersion)
import time
print(time.asctime())

filen='/scratch/RDKit_git/Data/Zinc/zinc_all_clean.pkl.gz'

import copy
history={} # we will use this to see how quickly the results converge
counts=defaultdict(lambda:defaultdict(int))
t1 = time.time()
with gzip.open(filen,'rb') as inf:
    i = 0
    while 1:
        try:
            m,nm = pickle.load(inf)
        except EOFError:
            break
        if not m: continue
        i+=1
        for v in 1,2,3:
            onbits=len(rdmd.GetMorganFingerprint(m,v).GetNonzeroElements())
            counts[(v,-1)][onbits]+=1
            for l in 1024,2048,4096,8192:
                dbits = onbits-rdmd.GetMorganFingerprintAsBitVect(m,v,l).GetNumOnBits()
                counts[(v,l)][dbits]+=1
        if not i%20000:
            t2 = time.time()
            print("Done %d in %.2f sec"%(i,t2-t1))
        if not i%100000:
            history[i] = copy.deepcopy(counts)

pickle.dump(dict(counts),gzip.open('../data/fp_collision_counts.pkl.gz','wb+'))

for k,d in history.items():
    history[k] = dict(d)
pickle.dump(dict(history),gzip.open('../data/fp_collision_counts.history.pkl.gz','wb+'))

figure(figsize=(16,20))

pidx=1
#----------------------------
for nbits in (8192,4096,2048,1024):
    subplot(4,2,pidx)
    pidx+=1
    maxCollisions = max(counts[3,nbits].keys())+1
    d1=np.zeros(maxCollisions,np.int)
    for k,v in counts[1,nbits].items():
        d1[k]=v
    d2=np.zeros(maxCollisions,np.int)
    for k,v in counts[2,nbits].items():
        d2[k]=v
    d3=np.zeros(maxCollisions,np.int)
    for k,v in counts[3,nbits].items():
        d3[k]=v
    barWidth=.25
    locs = np.array(range(maxCollisions))

    bar(locs,d1,barWidth,color='b',label="r=1")
    bar(locs+barWidth,d2,barWidth,color='g',label="r=2")
    bar(locs+2*barWidth,d3,barWidth,color='r',label="r=3")
    
    #_=hist((d1,d2,d3),bins=20,log=True,label=("r=1","r=2","r=3"))
    title('%d bits'%nbits)
    _=yscale("log")
    _=legend()

    subplot(4,2,pidx)
    pidx+=1
    plot([x for x,y in counts[1,-1].items()],[y for x,y in counts[1,-1].items()],'.b-',label=
        "r=1")
    plot([x for x,y in counts[2,-1].items()],[y for x,y in counts[2,-1].items()],'.g-',label=
        "r=2")
    plot([x for x,y in counts[3,-1].items()],[y for x,y in counts[3,-1].items()],'.r-',label=
        "r=3")
    _=xlabel("num bits set")
    _=ylabel("count")
    _=legend()
    
    

figure(figsize=(16,20))

pidx=1
#----------------------------
for nbits in (8192,4096,2048,1024):
    subplot(4,2,pidx)
    pidx+=1
    maxCollisions = max(counts[3,nbits].keys())+1
    d1=np.zeros(maxCollisions,np.int)
    for k,v in counts[1,nbits].items():
        d1[k]=v
    d2=np.zeros(maxCollisions,np.int)
    for k,v in counts[2,nbits].items():
        d2[k]=v
    d3=np.zeros(maxCollisions,np.int)
    for k,v in counts[3,nbits].items():
        d3[k]=v
    barWidth=.25
    locs = np.array(range(maxCollisions))

    bar(locs,d1,barWidth,color='b',label="r=1")
    bar(locs+barWidth,d2,barWidth,color='g',label="r=2")
    bar(locs+2*barWidth,d3,barWidth,color='r',label="r=3")
    
    #_=hist((d1,d2,d3),bins=20,log=True,label=("r=1","r=2","r=3"))
    title('%d bits'%nbits)
    _=yscale("log")
    _=legend()

    subplot(4,2,pidx)
    pidx+=1
    plot([x for x,y in counts[1,-1].items()],[y for x,y in counts[1,-1].items()],'.b-',label=
        "r=1")
    plot([x for x,y in counts[2,-1].items()],[y for x,y in counts[2,-1].items()],'.g-',label=
        "r=2")
    plot([x for x,y in counts[3,-1].items()],[y for x,y in counts[3,-1].items()],'.r-',label=
        "r=3")
    _=xlabel("num bits set")
    _=ylabel("count")
    _=legend()
    
    

