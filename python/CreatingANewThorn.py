import os
os.environ["PATH"]="/opt/conda/envs/python2/bin:"+os.environ["PATH"]

get_ipython().system('python --version')

get_ipython().magic('cd ~/CactusFW2')

# Define some basic parameters describing a new thorn
thorn_pars = {
   "thorn_name" : "EnergyCalc",
   "arrangement_name" : "FunwaveUtils",
   "author" : "Steven R. Brandt",
   "email" : "sbrandt@cct.lsu.edu",
   "license" : "BSD"
 }
import os
os.environ["ARR"]=thorn_pars["arrangement_name"]
os.environ["THORN"]=thorn_pars["thorn_name"]

get_ipython().system('rm -fr arrangements/$ARR/$THORN')

import re
# This function does substitutes all occurances of "{name}" in input_str with values["name"]
# and returns the new string.
def replace_values(input_str,values):
    while True:
        g = re.search(r'{(\w+)}',input_str)
        if g:
            var = g.group(1)
            if var in values:
                val = values[var]
            else:
                raise Exception("Undefined: <<"+var+">>")
            start = g.start(0)
            end = start+len(g.group(0))
            input_str = input_str[0:start]+val+input_str[end:]
            continue
        break
    return input_str

# Createas a file with the given name, uses replace_values()
# to update file_contents with file_values, and writes it out.
def create_file(file_name,file_contents,file_values):
    fd = open(file_name,"w")
    file_contents = replace_values(file_contents,file_values)
    print("Over-writing file '"+file_name+"'")
    file_contents = re.sub(r'^\s+','',file_contents)
    fd.write(file_contents)
    fd.close()
    
# The equivalent of mkdir -p
def create_dir(dir):
    print("Ensuring directory '"+dir+"'")
    if not os.path.exists(dir):
        os.makedirs(dir)

interface_ccl_contents = """
## Interface definitions for thorn {thorn_name}
inherits:
## An implementation name is required for all thorns. No
## two thorns in a configuration can implement the same
## interface.
implements: {thorn_name}

## the groups declared below can be public, private, or protected.
public:

## A group defines a set of variables that are allocated together
## and share common properties, i.e. timelevels, tags such as the
## Prolongation=None tag. The type tag can take on the values
## GF, Scalar, or Array.

## Note that the number of timelevels can be an integer parameter
## GF stands for "Grid Function" and refers to a distributed array
## data structure.
#cctk_real force_group type=GF timelevels=3 tags='Prolongation="None"'
#{
#  force1, force2
#}

## Scalars are single variables that are available on all processors.
#cctk_real scalar_group type=SCALAR 
#{
#  scalar1, scalar2
#}
"""

schedule_ccl_contents = """
## Schedule definitions for thorn {thorn_name}

## There won't be any storage allocated for a group
## unless a corresponding storage declaration exists
## for it in the schedule file. In square brackets,
## we specify the number of storage levels to allocate.
## These commented examples correspond to the commented
## examples in the interface file above.
# storage: force_group[3]
# storage: scalar_group

## Schedule a function defined in this thorn to run at the beginning
## of the simulation. The minimum you need to specify for a schedule
## item is what language it's written in. Choices are: C (which includes
## C++) and Fortran (which means Fortran90).
#SCHEDULE init_function at CCTK_INIT
#{
#  LANG: C
#}"Do some initial stuff"
"""

param_ccl_contents = """
## Parameter definitions for thorn {thorn_name}
## There are five types of parameters: int, real, keyword, string, and boolean.
## The comments provide prototypes of each.
#
#CCTK_INT one_to_five "This integer parameter goes from 1 to 5"
#{
#  1:5 :: "Another comment"
#} 3 # This is the default value
#
#CCTK_REAL from_2p5_to_3p8e4 "This integer parameter goes from 2.5 to 3.8e4"
#{
#  2.5:3.8e4 :: "Another comment"
#} 4.4e3 # This is the default value
#
## This keyword example defines the parameter wavemaker_type and 8 possible values.
#CCTK_KEYWORD wavemaker_type "types of wave makers"
#{
#  "ini_rec" :: "initial rectangular hump, need xc,yc and wid"
#  "lef_sol" :: "initial solitary wave, WKN B solution, need amp, dep"
#  "ini_oth" :: "other initial distribution specified by users"
#  "wk_reg" :: "Wei and Kirby 1999 internal wave maker, need xc_wk, tperiod, amp_wk, dep_wk, theta_wk, and time_ramp (factor of period)"
#  "wk_irr" :: "Wei and Kirby 1999 TMA spectrum wavemaker, need xc_wk, dep_wk, time_ramp, delta_wk, freqpeak, freqmin, freqmax, hmo, gammatma, theta-peak"
#  "wk_time_series" :: "fft a time series to get each wave component and then use Wei and Kirby's ( 1999) wavemaker. Need input wavecompfile (including 3 columns:   per,amp,pha) and numwavecomp, peakperiod, dep_wk, xc_wk, ywidth_wk"
#  "ini_gau" :: "initial Gaussian hump, need amp, xc, yc, and wid"
#  "ini_sol" :: "initial solitary wave, xwavemaker"
#}"wk_reg" # This is the default value
#
#CCTK_STRING a_string_par "a comment"
#{
#  .* :: "This is a perl 5 regular expression defining what the string may contain"
#} "blah blah blah" # This is the default value
#
#BOOLEAN a_boolean_par "a comment"
#{
#} true


"""

configuration_ccl_contents = """
# Configuration definitions for thorn {thorn_name}
## You should not need include "mpi.h", but if you
## do, you will need this next line.
# REQUIRES MPI
# REQUIRES HDF5
"""

makefile_contents = """
# Main make.code.defn file for thorn {thorn_name}

# Source files in this directory
SRCS =

# Subdirectories containing source files
SUBDIRS =
"""

readme_contents = """
Author(s)    : {author} <{email}>
Maintainer(s): {author} <{email}>
Licence      : {license}
--------------------------------------------------------------------------

1. Purpose

not documented
"""

import os

# This function will create a complete thorn
def create_thorn():
  # Create the thorn directory inside the Cactus source tree
  arrangement_dir = "arrangements/"+thorn_pars["arrangement_name"]
  thorn_dir = arrangement_dir + "/" + thorn_pars["thorn_name"]
  create_dir(thorn_dir)

  # Create basic ccl files needed by Cactus
  schedule_ccl = thorn_dir+"/schedule.ccl"
  interface_ccl = thorn_dir+"/interface.ccl"
  param_ccl = thorn_dir+"/param.ccl"
  configuration_ccl = thorn_dir+"/configuration.ccl"
  create_file(schedule_ccl,schedule_ccl_contents,thorn_pars)
  create_file(interface_ccl,interface_ccl_contents,thorn_pars)
  create_file(param_ccl,param_ccl_contents,thorn_pars)
  create_file(configuration_ccl,configuration_ccl_contents,thorn_pars)

  # Create the source directory and first makefile
  src_dir = thorn_dir+"/src"
  create_dir(src_dir)
  create_file(src_dir+"/make.code.defn",makefile_contents,thorn_pars)

  # Other dirs and files not strictly needed
  # for compiling and running Cactus.
  create_file(thorn_dir+"/README",readme_contents,thorn_pars)
  
  test_dir = thorn_dir+"/test"
  create_dir(test_dir)
  par_dir = thorn_dir+"/par"
  create_dir(par_dir)
  doc_dir = thorn_dir+"/doc"
  create_dir(doc_dir)

create_thorn()

my_thorns_contents="""
# ./configs/sim/ThornList
# This file was automatically generated using the GetComponents script.

!CRL_VERSION = 2.0


# Component list: funwave.th

!DEFINE ROOT = CactusFW2
!DEFINE ARR = $ROOT/arrangements
!DEFINE ET_RELEASE = trunk
!DEFINE FW_RELEASE = FW_2014_05

#Cactus Flesh
!TARGET   = $ROOT
!TYPE     = git
!URL      = https://bitbucket.org/cactuscode/cactus.git
!NAME     = flesh
!CHECKOUT = CONTRIBUTORS COPYRIGHT doc lib Makefile src

!TARGET   = $ARR
!TYPE     = git
!URL      = https://bitbucket.org/stevenrbrandt/cajunwave.git
!REPO_PATH= $2
# Old version
#!AUTH_URL = https://svn.cct.lsu.edu/repos/projects/ngchc/code/branches/$FW_RELEASE/$1/$2
#!URL = https://svn.cct.lsu.edu/repos/projects/ngchc/code/branches/$FW_RELEASE/$2
#!URL = https://svn.cct.lsu.edu/repos/projects/ngchc/code/CactusCoastal/$2
!CHECKOUT =
CactusCoastal/Funwave
CactusCoastal/FunwaveMesh
CactusCoastal/FunwaveCoord
CactusCoastal/Tridiagonal
CactusCoastal/Tridiagonal2

# CactusBase thorns
!TARGET   = $ARR
!TYPE     = git
!URL      = https://bitbucket.org/cactuscode/cactusbase.git
!REPO_PATH= $2
!CHECKOUT =
CactusBase/Boundary
CactusBase/CartGrid3D
CactusBase/CoordBase
CactusBase/Fortran
CactusBase/InitBase
CactusBase/IOASCII
CactusBase/IOBasic
CactusBase/IOUtil
CactusBase/SymBase
CactusBase/Time
#
# CactusNumerical thorns
!TARGET   = $ARR
!TYPE     = git
!URL      = https://bitbucket.org/cactuscode/cactusnumerical.git
!REPO_PATH= $2
!CHECKOUT =
!CHECKOUT =
CactusNumerical/MoL
CactusNumerical/LocalInterp

CactusNumerical/Dissipation
CactusNumerical/SpaceMask
CactusNumerical/SphericalSurface
CactusNumerical/LocalReduce
CactusNumerical/InterpToArray

!TARGET   = $ARR
!TYPE     = git
!URL      = https://bitbucket.org/cactuscode/cactusutils.git
!REPO_PATH= $2
!CHECKOUT = CactusUtils/Accelerator CactusUtils/OpenCLRunTime
CactusUtils/NaNChecker
CactusUtils/Vectors
CactusUtils/SystemTopology

# Carpet, the AMR driver
!TARGET   = $ARR
!TYPE     = git
!URL      = https://bitbucket.org/eschnett/carpet.git
!REPO_PATH= $2
!CHECKOUT = Carpet/doc
Carpet/Carpet
Carpet/CarpetEvolutionMask
Carpet/CarpetIOASCII
Carpet/CarpetIOBasic
Carpet/CarpetIOHDF5
Carpet/CarpetIOScalar
#Carpet/CarpetIntegrateTest
Carpet/CarpetInterp
Carpet/CarpetInterp2
Carpet/CarpetLib
Carpet/CarpetMask
#Carpet/CarpetProlongateTest
Carpet/CarpetReduce
Carpet/CarpetRegrid
Carpet/CarpetRegrid2
#Carpet/CarpetRegridTest
Carpet/CarpetSlab
Carpet/CarpetTracker
Carpet/CycleClock
#Carpet/HighOrderWaveTest
Carpet/LoopControl
#Carpet/ReductionTest
#Carpet/ReductionTest2
#Carpet/ReductionTest3
#Carpet/RegridSyncTest
Carpet/TestCarpetGridInfo
Carpet/TestLoopControl
Carpet/Timers

# Additional Cactus thorns
!TARGET   = $ARR
!TYPE     = svn
!URL      = https://svn.cactuscode.org/projects/$1/$2/trunk
!CHECKOUT = ExternalLibraries/OpenBLAS ExternalLibraries/OpenCL ExternalLibraries/pciutils ExternalLibraries/PETSc
ExternalLibraries/MPI
ExternalLibraries/HDF5
ExternalLibraries/zlib
ExternalLibraries/hwloc

# Simulation Factory
!TARGET   = $ROOT/simfactory
!TYPE     = git
!URL      = https://bitbucket.org/simfactory/simfactory2.git
!NAME     = simfactory2
!CHECKOUT = README.md README_FIRST.txt bin doc etc lib mdb

# Various thorns from LSU
#!TARGET   = $ARR
#!TYPE     = git
#!URL      = https://bitbucket.org/einsteintoolkit/archivedthorns-vectors.git
#!REPO_PATH= $2
#!CHECKOUT =
#LSUThorns/Vectors
#LSUThorns/QuasiLocalMeasures
#LSUThorns/SummationByParts
#LSUThorns/Prolong

#Roland/MapPoints
#Tutorial/BadWaveMoL
#Tutorial/BasicWave
#Tutorial/BasicWave2
#Tutorial/BasicWave3

# Various thorns from the AEI
# Numerical
!TARGET   = $ARR
!TYPE     = git
!URL      = https://bitbucket.org/cactuscode/numerical.git
!REPO_PATH= $2
!CHECKOUT =
#AEIThorns/ADMMass
AEIThorns/AEILocalInterp
#AEIThorns/PunctureTracker
#AEIThorns/SystemStatistics
#AEIThorns/Trigger
{arrangement_name}/{thorn_name}"""
create_file("my_thorns.th",my_thorns_contents,thorn_pars)

get_ipython().system('time ./simfactory/bin/sim build -j 2 --thornlist=./my_thorns.th')



