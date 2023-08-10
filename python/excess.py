get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np, pandas as pd
from math import sqrt, pi, exp
import os.path, os, sys, json
try:
    workdir
except NameError:
    workdir = get_ipython().magic('pwd')
else:
    get_ipython().magic('cd -q $workdir')
print(workdir)

get_ipython().run_cell_magic('bash', '-s "$workdir"', '%cd -q $1\n\necho \'fau_example(excess "./" excess.cpp)\' > mc/CMakeLists.txt\n\nif [ ! -d "faunus/" ]; then\n    git clone https://github.com/mlund/faunus.git\n    cd faunus\n    git checkout master 7d022a78c67dd46c5ac32aa7f06e00b08459a1db\nelse\n    cd faunus\nfi\n\ncmake . -DCMAKE_BUILD_TYPE=Release -DENABLE_APPROXMATH=on -DMYPLAYGROUND=$1/mc &>/dev/null\nmake excess -j4\n%cd $1')

# definition of salts (n=stoichiometry; z=valency; r=radius; L=box length; activities in mol/l)
salts = pd.Series(
    {
        'NaCl' : pd.Series(
            dict(ion1='Na', ion2='Cl', n1=1, n2=1, z1=1, z2=-1, r1=2.2, r2=2.2, L=50,
                 activities=np.arange(0.1,1.6,0.05), exp='exp-nacl-coeff.csv',
                 color='red', label=u'NaCl' ) ),
        'Na3Cit' : pd.Series(
            dict(ion1='Na', ion2='Cit', n1=3, n2=1, z1=1, z2=-3, r1=2.3, r2=3.5, L=100,
                 activities=np.arange(0.005,0.1,0.005), exp='exp-na3cit-coeff.csv',
                 color='blue', label=u'Na$_3$(C$_6$H$_5$O$_7$)' ) ),
        'NaAc' : pd.Series(
            dict(ion1='Na', ion2='Ac', n1=1, n2=1, z1=1, z2=-1, r1=2.3, r2=2.95, L=50,
                 activities=np.arange(0.1,1.6,0.05), exp='exp-naac-coeff.csv',
                 color='green', label=u'Na(CH$_3$COO)' ) ),
        'NH4Cl' : pd.Series(
            dict(ion1='NH4', ion2='Cl', n1=1, n2=1, z1=1, z2=-1, r1=1.75, r2=2.3, L=50,
                 activities=np.arange(0.1,1.6,0.05), exp='exp-nh4cl-coeff.csv',
                 color='magenta', label=u'NH$_4$Cl' ) ),
        'GdnCl' : pd.Series(
            dict(ion1='Gdn', ion2='Cl', n1=1, n2=1, z1=1, z2=-1, r1=0.55, r2=2.3, L=50,
                 activities=np.arange(0.1,1.6,0.05), exp='exp-gdncl-coeff.csv',
                 color='black', label=u'GdnCl' ) )
    }
)

get_ipython().magic('cd -q $workdir')

def mkinput():
    js = {
            "moleculelist": {
                "salt": { "Ninit": 20, "atomic": True, "atoms": d.n1*(d.ion1+' ') + d.n2*(d.ion2+' ') }
            }, 
            "energy": {
                "nonbonded": { "coulomb": { "epsr": 80 } }
            }, 
            "moves": {
                "atomtranslate": { "salt": { "permol": True, "prob": 0.01 } }, 
                "atomgc": { "molecule": "salt" }
            }, 
            "system": {
                "mcloop"      : { "macro": 10, "micro": micro }, 
                "cuboid"      : { "len": d.L },
                "coulomb"     : { "epsr": 80 },
                "temperature" : 298.15
            },
            "analysis": {
                "pqrfile"   : { "file" : "confout.pqr" },
                "statefile" : { "file" : "state"}            
            },
            "atomlist": {
                d.ion1: { "q": d.z1, "r": d.r1, "eps":0.01, "dp": 50, "activity": d.n1*activity }, 
                d.ion2: { "q": d.z2, "r": d.r2, "eps":0.01, "dp": 50, "activity": d.n2*activity }
            }
    }

    with open('excess.json', 'w+') as f:
        f.write(json.dumps(js, indent=4))

# flow control:
equilibration = True   # if true, delete state file and start over
production    = True   # if true, start from present state file
override      = False  # if true, override existing files

for salt, d in salts.iteritems():
    for activity in d.activities:
        
        pfx=salt+'-a'+str(activity)
        if not os.path.isdir(pfx):
            get_ipython().magic('mkdir -p $pfx')
        else:
            if override==False:
                break
        get_ipython().magic('cd $pfx')

        # equilibration run
        if equilibration:
            get_ipython().system('rm -fR state')
            micro=5000
            mkinput()
            get_ipython().system('../mc/excess > eq 2>&1')
            get_ipython().system('rm -fR analysis_out.json')

        # production run
        if production:
            micro=50000
            mkinput()
            get_ipython().system('../mc/excess > out 2>&1')

        get_ipython().magic('cd -q ..')

print('done.')

get_ipython().magic('cd -q $workdir')
import json

for salt, d in salts.iteritems():
    d['g1'] = []
    d['g2'] = []
    for activity in d.activities:
        pfx=salt+'-a'+str(activity)

        if os.path.isdir(pfx):
            get_ipython().magic('cd -q $pfx')
            r = json.load( open('move_out.json') )['moves']['Grand Canonical Salt']['atoms']
            d['g1'].append( r[d.ion1]['gamma'] )
            d['g2'].append( r[d.ion2]['gamma'] )
            get_ipython().magic('cd -q ..')
salts.NaCl

get_ipython().magic('cd -q $workdir')
def ionicstrength(c, n1, n2, z1, z2):
    return 0.5*( n1*c*z1**2 + n2*c*z2**2 )

def meanactivity(gA, gB, p, q):
    ''' mean ionic activity coefficient'''
    return ( gA**p * gB**q ) ** (1.0/(p+q))

for salt, d in salts.iteritems():
    # experiment (already tabulated as ionic strength)
    I, g = np.loadtxt(d.exp, delimiter=',', skiprows=1, unpack=True)
    plt.plot(I, g, marker='o', ls='none', ms=8, mew=0, color=d.color)
    
    # simulation
    g = meanactivity( np.array(d.g1), np.array(d.g2), d.n1, d.n2 )
    C = d.activities / g # molarity
    I = ionicstrength(C, d.n1, d.n2, d.z1, d.z2)
    
    plt.plot( I, g, label=d.label, lw=3, color=d.color)

plt.legend(loc=0, frameon=False, ncol=3)
plt.ylabel('Mean activity coefficient, $\gamma_{\pm}$')
plt.xlabel('Ionic strength (mol/l)')
plt.xlim((0.04,1.25))
plt.ylim((0,0.9))
plt.title('Experiment (symbols) vs simulation (lines)')
print('')

from scipy.optimize import newton

lB=7           # Bjerrum length for water (Å)
s=2*2.2        # sigma (Å)
eps=0.01 / 2.5 # epsilon (kT)
r = np.linspace(0.64*s, 2*s, 100)

def el(r): return -lB/r                        # Coulomb pot.
def lj(r): return 4*eps*( (s/r)**12-(s/r)**6 ) # Lennard-Jones pot.
def u(r): return lj(r) + el(r)                 # combined pot.

plt.plot(r, lj(r), 'r.', label='LJ', markersize=4 )
plt.plot(r, el(r), label='Coulomb', lw=2)
plt.plot(r, u(r), label='LJ+Coulomb', lw=2 )
plt.xlabel(u'$r$/Å')
plt.ylabel(u'$u(r)/k_BT$')
plt.legend(loc=0, frameon=False)

r0 = newton(u, x0=0.5*s) # when is u(r) zero?
plt.plot([r0], [0], 'go')
plt.annotate( '$r_0=$'+str('%.1f' % r0)+u' Å', xy=(r0, 0), xytext=(r0+1, -0.8),
             arrowprops=dict(facecolor='green', width=2, headwidth=6, edgecolor='none', shrink=0.1) )

plt.plot([s], [0], 'ro')
plt.annotate( '$r_0=$'+str('%.1f' % s)+u' Å', xy=(s, 0), xytext=(s+1, 0.8),
             arrowprops=dict(facecolor='red', width=2, headwidth=6, edgecolor='none', shrink=0.1) )

def mkinput():
    global micro, d, activity, macromolecule, pH
    js = {
            "moleculelist": {
                "_protein": { "structure":macromolecule, "Ninit":1, "insdir":"0 0 0" },
                "counter": { "Ninit": 8, "atomic": True, "atoms":"CNa" },
                "salt":    { "Ninit": 30, "atomic": True, "atoms": d.n1*(d.ion1+' ') + d.n2*(d.ion2+' ') }
            },
            "energy": {
                "nonbonded": { "coulomb": { "epsr": 80 } }
            },
            "moves": {
                "atomtranslate": {
                    "salt": { "peratom": True, "prob": 1.0 },
                    "counter": { "peratom": True, "prob": 1.0 }
                }
            },
            "system": {
                "mcloop"      : { "macro": 10, "micro": micro }, 
                "cuboid"      : { "len": d.L },
                "coulomb"     : { "epsr": 80 },
                "temperature" : 298.15
            },
            "atomlist": {
                d.ion1 :  { "q": d.z1, "r": d.r1, "eps":0.01, "dp": 50, "activity": d.n1*activity }, 
                d.ion2 :  { "q": d.z2, "r": d.r2, "eps":0.01, "dp": 50, "activity": d.n2*activity },
                "CNa"  :  { "q": d.z1, "r": d.r1, "eps":0.01, "dp": 50 },
                "ASP"  :  { "q":-1, "r":3.6, "mw":110, "eps":0.01 },
                "HASP" :  { "q":0,  "r":3.6, "mw":110, "eps":0.01 },
                "LASP" :  { "q":2,  "r":3.6, "mw":110, "eps":0.01 },
                "CTR"  :  { "q":-1, "r":2.0, "mw":16, "eps":0.01 },
                "HCTR" :  { "q":0,  "r":2.0, "mw":16, "eps":0.01 },
                "GLU"  :  { "q":-1, "r":3.8, "mw":122, "eps":0.01 },
                "HGLU" :  { "q":0,  "r":3.8, "mw":122, "eps":0.01 },
                "LGLU" :  { "q":2,  "r":3.8, "mw":122, "eps":0.01 },
                "HIS"  :  { "q":0,  "r":3.9, "mw":130, "eps":0.01 },
                "HHIS" :  { "q":1,  "r":3.9, "mw":130, "eps":0.01 },
                "NTR"  :  { "q":0,  "r":2.0, "mw":14, "eps":0.01 },
                "HNTR" :  { "q":1,  "r":2.0, "mw":14, "eps":0.01 },
                "TYR"  :  { "q":-1, "r":4.1, "mw":154, "eps":0.01 },
                "HTYR" :  { "q":0,  "r":4.1, "mw":154, "eps":0.01 },
                "LYS"  :  { "q":0,  "r":3.7, "mw":116, "eps":0.01 },
                "HLYS" :  { "q":1,  "r":3.7, "mw":116, "eps":0.01 },
                "CYb"  :  { "q":0,  "r":3.6, "mw":103, "eps":0.01 },
                "CYS"  :  { "q":-1, "r":3.6, "mw":103, "eps":0.01 },
                "HCYS" :  { "q":0,  "r":3.6, "mw":103, "eps":0.01 },
                "ARG"  :  { "q":0,  "r":4.0, "mw":144, "eps":0.01 },
                "HARG" :  { "q":1,  "r":4.0, "mw":144, "eps":0.01 },
                "ALA"  :  { "q":0,  "r":3.1, "mw":66, "eps":0.01 },
                "ILE"  :  { "q":0,  "r":3.6, "mw":102, "eps":0.01 },
                "LEU"  :  { "q":0,  "r":3.6, "mw":102, "eps":0.01 },
                "MET"  :  { "q":0,  "r":3.8, "mw":122, "eps":0.01 },
                "PHE"  :  { "q":0,  "r":3.9, "mw":138, "eps":0.01 },
                "PRO"  :  { "q":0,  "r":3.4, "mw":90, "eps":0.01 },
                "TRP"  :  { "q":0,  "r":4.3, "mw":176, "eps":0.01 },
                "VAL"  :  { "q":0,  "r":3.4, "mw":90, "eps":0.01 },
                "SER"  :  { "q":0,  "r":3.3, "mw":82, "eps":0.01 },
                "THR"  :  { "q":0,  "r":3.5, "mw":94, "eps":0.01 },
                "ASN"  :  { "q":0,  "r":3.6, "mw":108, "eps":0.01 },
                "GLN"  :  { "q":0,  "r":3.8, "mw":120, "eps":0.01 },
                "GLY"  :  { "q":0,  "r":2.9, "mw":54, "eps":0.01 }
            },
            "analysis": {
                "xtcfile"   : { "file" : "traj.xtc", "nstep":10 },
                "pqrfile"   : { "file" : "confout.pqr" },
                "aamfile"   : { "file" : "confout.aam" },
                "statefile" : { "file" : "state" }            
            }
    }
    
    if (TITRATION):
        js['moves']['gctit'] = {
            "molecule": "salt", "prob": 1.0,
            "processes" : {
                "H-Asp" : { "bound":"HASP", "free":"ASP", "pKd":4.0, "pX":pH },
                "H-Ctr" : { "bound":"HCTR", "free":"CTR", "pKd":2.6, "pX":pH },
                "H-Glu" : { "bound":"HGLU", "free":"GLU", "pKd":4.4, "pX":pH },
                "H-His" : { "bound":"HHIS", "free":"HIS", "pKd":6.3, "pX":pH },
                "H-Arg" : { "bound":"HARG", "free":"ARG", "pKd":12.0, "pX":pH },
                "H-Ntr" : { "bound":"HNTR", "free":"NTR", "pKd":7.5, "pX":pH },
                "H-Cys" : { "bound":"HCYS", "free":"CYS", "pKd":10.8, "pX":pH },
                "H-Tyr" : { "bound":"HTYR", "free":"TYR", "pKd":9.6, "pX":pH },
                "H-Lys" : { "bound":"HLYS", "free":"LYS", "pKd":10.4,"pX":pH }
            }
        }
    if (GC):
        js['moves']['atomgc'] = { "molecule": "salt" }

    print('Writing JSON file')
    with open('excess.json', 'w') as f:
        f.write(json.dumps(js, indent=4))
        
def WriteAAM( filename, aam ):
    f = open( filename,'w' )
    f.write( str(len(aam)) + '\n' )
    for i,j in aam.iterrows():
        f.write('{0:4} {1:5} {2:8.3f} {3:8.3f} {4:8.3f} {5:6.3f} {6:6.2f} {7:6.2f}\n'                .format(j['name'], i+1, j.x, j.y, j.z, j.q, j.mw, j.r))
    f.close()

def AverageChargeAAM(aamfile, gctit_output, group='_protein', scale2int=True):
    from math import floor
    ''' Save aam file w. average charges based on gctit-output (json file) '''
    aam = pd.read_table(aamfile, sep=' ', skiprows=1, usecols=[0,2,3,4,5,6,7], names=['name','x','y','z','q','mw','r'])

    js = json.load( open(gctit_output) ) [group]
    js = pd.DataFrame( js, index=js['index'] )

    f = 1.0
    if scale2int:
        total = js.charge.sum()
        target = round(total, 0)
        f = target / total
    
    aam.q = f*js.charge        # copy avg. charges of titratable sites
    aam.q = aam.q.fillna(0)  # fill missing
    
    print('net charge = ', aam.q.sum())
            
    return aam

get_ipython().magic('cd $workdir')
macromolecule = '../hsa-ph7.aam'
xtc_freq      = 1e200      # frequency to save xtc file (higher=less frames)
pH_range      = np.arange(6, 8, 1).tolist()
activity      = 0.1        # mol/l
d             = salts.NaCl # which salt
d.L           = 120        # box length (Å)
TITRATION     = True       # Do pH titration
GC            = False

tdata         = { 'pH': [], 'Z': [] } # store titration curve here

salt = d.label

def getPrefix( salt, pH, activity ):
    return str(salt) + '-titration-pH' + str(pH) + '-activity' + str(activity)

for pH in pH_range:
    pfx = getPrefix(salt, pH, activity)
    
    if not os.path.isdir(pfx): # run simulation
        get_ipython().magic('mkdir -p $pfx')
        get_ipython().magic('cd $pfx')

        # equilibration run (no translation)
        get_ipython().system('rm -fR state')
        micro=500
        mkinput()
        get_ipython().system('../mc/excess > eq.tit')

        # production run
        micro=2000
        mkinput()
        get_ipython().system('../mc/excess > out.tit')
        
        # create an aam file w. average charges
        aam = AverageChargeAAM( macromolecule, 'gctit-output.json', group='_protein', scale2int=True )
        WriteAAM( 'avgcharge.aam', aam )
        get_ipython().magic('cd -q ..')
        
    if os.path.isdir(pfx): # analysis
        get_ipython().magic('cd -q $pfx')
        aam = AverageChargeAAM( macromolecule, 'gctit-output.json', group='_protein', scale2int=False )
        tdata['pH'].append( pH )
        tdata['Z'].append( aam.q.sum() )
        get_ipython().magic('cd -q ..')

if len( tdata['pH'] ) > 0:
    plt.plot(tdata['pH'], tdata['Z'], 'bo')
    plt.xlabel('pH')
    plt.ylabel('Z')
    plt.title(u'pH titration curve ($a_{NaCl}=$'+str(activity)+')')

print('done.')

TITRATION = False

import mdtraj as md

activity=0.3   # mol/l
d = salts.NaCl # which salt
d.L=120        # box length (Å)

eq=True
prod=True
analyse=True

get_ipython().magic('cd $workdir')

for pH in pH_range:
    pfx = getPrefix(salt, pH, activity)
    if os.path.isdir(pfx):

        get_ipython().magic('cd $pfx')

        # make gromacs index file for selecting ions
        f = open('index.ndx', 'w')
        traj = md.load('confout.gro')
        ndx_ion1 = traj.top.select('name '+d.ion1) # index w. all cations
        ndx_ion2 = traj.top.select('name '+d.ion2) # index w. all anions
        ndx_sys  = traj.top.select('all')
        f.write( '[' + d.ion1 + ']\n' + ' '.join(str(e+1) for e in ndx_ion1 ) + 2*'\n')
        f.write( '[' + d.ion2 + ']\n' + ' '.join(str(e+1) for e in ndx_ion2 ) + 2*'\n')
        f.write( '[ System ]\n' + ' '.join(str(e+1) for e in ndx_sys ) + 2*'\n')
        f.close()

        if eq:
            print('Equilibration run - GC enabled')
            GC=True
            micro=500
            mkinput()
            get_ipython().system('rm -fR state')
            get_ipython().system('$workdir/mc/excess > eq.gc')

        if prod:
            print('Production run - GC disabled')
            GC=False
            micro=10000
            mkinput()
            get_ipython().system('$workdir/mc/excess > out.nogc')

        if analyse:
            get_ipython().system('cp confout.pqr confout.pdb # dirty conversion, but will work for this purpose')
            get_ipython().system('echo -ne "$d.ion1\\nSystem" | gmx --quiet spatial -s confout.gro -f traj.xtc -bin 0.2 -nopbc -n index.ndx   # Sodium')
            get_ipython().system('mv grid.cube ion1.cube')
            get_ipython().system('echo -ne "$d.ion2\\nSystem" | gmx --quiet spatial -s confout.gro -f traj.xtc -bin 0.2 -nopbc -n index.ndx   # Chloride')
            get_ipython().system('mv grid.cube ion2.cube')
            get_ipython().system('rm confout.pdb')

        get_ipython().magic('cd -q ..')
print('done.')



