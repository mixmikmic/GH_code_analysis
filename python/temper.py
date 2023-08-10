from __future__ import division, unicode_literals, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np, pandas as pd
import os.path, os, sys, json
plt.rcParams.update({'font.size': 16, 'figure.figsize': [8.0, 6.0]})
try:
    workdir
except NameError:
    workdir = get_ipython().magic('pwd')
else:
    get_ipython().magic('cd $workdir')

get_ipython().run_cell_magic('bash', '-s "$workdir"', 'cd "$1"\nif [ ! -d "faunus/" ]; then\n  git clone https://github.com/mlund/faunus.git\n  cd faunus\n  git checkout 8eeef15b95e8fcabc85539a78153eb3f7d930874\nelse\n  cd faunus\nfi\n\n# if different, copy custom temper.cpp into faunus\nif ! cmp ../temper.cpp src/examples/temper.cpp >/dev/null 2>&1\nthen\n    cp ../temper.cpp src/examples/\nfi\n\nCXX=clang++ CC=clang cmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_APPROXMATH=on -DENABLE_MPI=on &>/dev/null\nmake example_temper -j4\n%cd ..')

b=1.75     # center-charge distance
radius=2.5 # hard-sphere radius
with open('cation.aam', 'w+') as f:
    f.writelines(
        ['2\n',
         'HS 1   0.0 0.0 0.0  0    1000   '+str(radius)+'\n',
         'POS 2  0.0 0.0 '+str(b)+' 1.0  0.01   0.0\n'])
    
with open('anion.aam', 'w+') as f:
    f.writelines(
        ['2\n',
         'HS 1   0.0 0.0 0.0  0    1000   '+str(radius)+'\n',
         'NEG 2  0.0 0.0 '+str(b)+' -1.0  0.01   0.0\n'])
f.close()

def mkinput(mpirank):
    """ function for creating a JSON input file for Faunus """
    js = {
        "atomlist" : {
            "HS" : { "dp":0, "q":0 },
            "NEG" : { "dp":0, "q":-1 },
            "POS" : { "dp":0, "q":1 }
        },
        "moleculelist" : {
            "cations" : { "structure":"cation.aam", "Ninit": N, "insdir":"1 1 1" },
            "anions" : { "structure":"anion.aam", "Ninit": N, "insdir":"1 1 1" }
        },
        "energy" : {
            "nonbonded" : {
                "coulomb" : { "epsr": epsr, "cutoff": 0.5*box }
             }
        },
        "moves" : {
            "moltransrot" : {
                "cations" : { "dp":0.5, "dprot":0.5, "dir":"1 1 1", "permol":True }, 
                "anions"  : { "dp":0.5, "dprot":0.5, "dir":"1 1 1", "permol":True } 
            },
            "temper" : { "format":"XYZ" }
        },
        "system" : {
            "temperature":298,
            "cuboid" : { "len" : box },
            "mcloop" : { "macro":macro, "micro":micro }
        }
    }

    with open('mpi'+str(mpirank)+'.temper.json', 'w+') as f:
        f.write(json.dumps(js, indent=4))

box=47.5   # cubic box side length
epsr=2     # dielectric cont.
macro=10   # number of macro loops
micro=10   # number of micro loops
N=400      # number of salt pairs

proclist=[0, 1, 2, 3]
epsrlist=[2, 3, 4, 5]

for i in proclist:
    epsr = epsrlist[i]
    mkinput(i)
    
get_ipython().system('mpirun -np 4 ./faunus/src/examples/temper')

for i in proclist:
    epsr = epsrlist[i]
    
    # load g(r) from disk
    r, g  = np.loadtxt('mpi'+str(i)+'.hs-hs.rdf', unpack=True)
    g[-1] = 2*g[-1] # correct for edge effects when binning
    
    # g(r) -> w(r) and shift to zero at long separations
    w = -np.log( g / r**2 )
    c = w[ r>23 ].mean()
    
    plt.plot(r,w-c, label=r'$\epsilon_r$='+str(epsr), lw=2)
    plt.xlabel(r'$r$/Å')
    plt.ylabel(r'$\beta w(r)+const$')
    plt.legend(loc=0, frameon=False)
    plt.title(r'$d=$'+str(2*radius)+' Å , $b=$'+str(b)+r' Å, $N=$'+str(N))



