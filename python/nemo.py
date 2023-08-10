from __future__ import division, unicode_literals, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
get_ipython().magic('matplotlib notebook')
import numpy as np, pandas as pd
import os.path, os, sys, json
from subprocess import call, check_output
from shutil import copyfile
try:
    workdir
except NameError:
    workdir = get_ipython().magic('pwd')
else:
    get_ipython().magic('cd $workdir')
print(workdir)

get_ipython().run_cell_magic('bash', '-s "$workdir"', 'cd $1\nif [ ! -d "faunus/" ]; then\n    echo \'fau_example(nemo "./" nemo.cpp)\' > mc/CMakeLists.txt\n    git clone https://github.com/bjornstenqvist/faunus.git\n    cd faunus\n    git checkout f9e6f969c0a82e75ab6facaf81b7bc6ce639a2ce\nelse\n  cd faunus\nfi\npwd\ncmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_APPROXMATH=on -DMYPLAYGROUND=`pwd`/../mc &>/dev/null\ncd `pwd`/../mc\nmake -j4')

get_ipython().run_cell_magic('time', '', 'def mkinput():\n    d = {\n     "atomlist" : {\n       "Na" : { "q": Zc, "sigma":1.0, "eps":0.0, "dp":0.1, "dprot":dprot }\n       },\n \n     "moleculelist" : {\n       "Na" : { "atoms":"Na", "Ninit":Nc, "atomic":True }\n       },\n \n     "energy" : {\n       "nonbonded" : { "coulomb" : { "epsr":1, "cutoff":10 } }\n       },\n \n     "moves" : {\n       "atomtranslate2Dhypersphere" : {\n         "Na" : { "peratom":True, "radius" :80 }\n         }\n       },\n \n     "system" : {\n       "temperature"   : 10,\n       "spheresurface" : { "radius" :Rs },\n       "mcloop"        : { "macro":10, "micro":micro },\n       "atomlist"      : "nemo.json",\n       "moleculelist"  : "nemo.json"\n            }\n        }\n    f = open(\'nemo.json\', \'w+\')\n    f.write(json.dumps(d, indent=4))\n    f.close()\n    \n# Parameters\nZc=1.0        # Value of each charge\nNc=100        # Number of charges\nRs=10.0       # Radius of the central, neutral sphere (Angstroms)\nmicro=1000    # Number of steps of the simulation /10\ndprot=300/Rs  # Displacement parameter of the charges\n\n%cd $workdir\'/mc\'    \n!rm -fR state\nmkinput()\n!./nemo > /dev/null')

from mpl_toolkits.mplot3d import Axes3D
x, y, z = np.genfromtxt('confout.pqr', unpack=True, usecols=(5,6,7), invalid_raise=False, skip_footer=1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.axis('off')
ax.scatter(x,y,z)

