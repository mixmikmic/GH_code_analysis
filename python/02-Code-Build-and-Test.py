get_ipython().system('mkdir -p ~/agave')

get_ipython().run_line_magic('cd', '~/agave')

get_ipython().system('pip install --upgrade setvar')

import re
import os
import sys
from setvar import *
from time import sleep

# This cell enables inline plotting in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

writefile("input.txt","""
!INPUT FILE FOR FUNWAVE_TVD
  ! NOTE: all input parameter are capital sensitive
  ! --------------------TITLE-------------------------------------
  ! title only for log file
TITLE = VESSEL
  ! -------------------HOT START---------------------------------
HOT_START = F
FileNumber_HOTSTART = 1
  ! -------------------PARALLEL INFO-----------------------------
  ! 
  !    PX,PY - processor numbers in X and Y
  !    NOTE: make sure consistency with mpirun -np n (px*py)
  !    
PX = 2
PY = 1
  ! --------------------DEPTH-------------------------------------
  ! Depth types, DEPTH_TYPE=DATA: from depth file
  !              DEPTH_TYPE=FLAT: idealized flat, need depth_flat
  !              DEPTH_TYPE=SLOPE: idealized slope, 
  !                                 need slope,SLP starting point, Xslp
  !                                 and depth_flat
DEPTH_TYPE = FLAT
DEPTH_FLAT = 10.0
  ! -------------------PRINT---------------------------------
  ! PRINT*,
  ! result folder
RESULT_FOLDER = output/

  ! ------------------DIMENSION-----------------------------
  ! global grid dimension
Mglob = 500
Nglob = 100

  ! ----------------- TIME----------------------------------
  ! time: total computational time/ plot time / screen interval 
  ! all in seconds
TOTAL_TIME = 3.0
PLOT_INTV = 1.0
PLOT_INTV_STATION = 50000.0
SCREEN_INTV = 1.0
HOTSTART_INTV = 360000000000.0

WAVEMAKER = INI_GAU
AMP = 3.0
Xc = 250.0
Yc = 50.0
WID = 20.0

  ! -----------------GRID----------------------------------
  ! if use spherical grid, in decimal degrees
  ! cartesian grid sizes
DX = 1.0
DY = 1.0
  ! ----------------SHIP WAKES ----------------------------
VESSEL_FOLDER = ./
NumVessel = 2
  ! -----------------OUTPUT-----------------------------
ETA = T
U = T
V = T
""")

writefile("run.sh","""
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib
mkdir -p rundir
cd ./rundir
cp ../input.txt .
mpirun -np 2 ~/FUNWAVE-TVD/src/funwave_vessel
""")

if os.environ.get('USE_TUNNEL') == 'True': 
    # fetch the hostname and port of the reverse tunnel running in the sandbox 
    # so Agave can connect to our local sandbox
    get_ipython().system("echo $(ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null sandbox 'curl -s  http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'') > ngrok_url.txt  ")
    get_ipython().system("cat ngrok_url.txt | sed 's|^tcp://||'")
    get_ipython().system("cat ngrok_url.txt | sed 's|^tcp://||' | sed -r 's#(.*):(.*)#\\1#' > ngrok_host.txt")
    get_ipython().system("cat ngrok_url.txt | sed 's|^tcp://||' | sed -r 's#(.*):(.*)#\\2#' > ngrok_port.txt")

    # set the environment variables otherwise set when running in a training cluster
    os.environ['VM_PORT'] = readfile('ngrok_port.txt').strip()
    os.environ['VM_MACHINE'] = readfile('ngrok_host.txt').strip()
    os.environ['AGAVE_SYSTEM_HOST'] = readfile('ngrok_host.txt').strip()
    os.environ['AGAVE_SYSTEM_PORT'] = readfile('ngrok_port.txt').strip()
    get_ipython().system('echo "VM_PORT=$VM_PORT"')
    get_ipython().system('echo "VM_MACHINE=$VM_MACHINE"')
    setvar("VM_IPADDRESS=$(getent hosts ${VM_MACHINE}|cut -d' ' -f1)")

get_ipython().system('scp -o "StrictHostKeyChecking=no" -P $VM_PORT input.txt run.sh $VM_IPADDRESS:.')
get_ipython().system('ssh -p $VM_PORT $VM_IPADDRESS bash run.sh')

get_ipython().system('ssh -p $VM_PORT $VM_IPADDRESS tar -C rundir -czf output.tgz output')
get_ipython().system('rm -fr output.tgz output')
get_ipython().system('scp -q -r -P $VM_PORT $VM_IPADDRESS:output.tgz .')
get_ipython().system('tar xzf output.tgz')

get_ipython().system('ls output')

data = np.genfromtxt("output/v_00003")
fig = plt.figure(figsize=(12,12))
pltres = plt.imshow(data[::-1,:])
plt.show()



