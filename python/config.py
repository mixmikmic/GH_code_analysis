get_ipython().system('mkdir -p ~/agave')

get_ipython().run_line_magic('cd', '~/agave')

get_ipython().system('pip3 install --upgrade setvar')

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

setvar("""
MACHINE_IP=$VM_IPADDRESS
MACHINE_NAME=nectar
DOMAIN=nectar.org
EMAIL=deardooley@gmail.com
AGAVE_USERNAME=dooley
MACHINE_USERNAME=jovyan
PORT=10022

DOCKERHUB_NAME=dooley
WORK_DIR=/home/${MACHINE_USERNAME}
HOME_DIR=/home/${MACHINE_USERNAME}
SCRATCH_DIR=/home/${MACHINE_USERNAME}
DEPLOYMENT_PATH=agave-deployment
AGAVE_JSON_PARSER=jq
AGAVE_TENANTS_API_BASEURL=https://agave-auth.solveij.com/tenants
APP_NAME=funwave-tvd-${MACHINE_NAME}-${AGAVE_USERNAME}
STORAGE_MACHINE=${MACHINE_NAME}-storage-${AGAVE_USERNAME}
EXEC_MACHINE=${MACHINE_NAME}-exec-${AGAVE_USERNAME}
REQUESTBIN_URL=$(requestbin-create)
""")

if os.environ.get('USE_TUNNEL') == "False" :
    get_ipython().system("ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null sandbox 'curl -s  http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'' > ngrok_url.txt")
    get_ipython().system("cat ngrok_url.txt | sed 's|^tcp://||' | sed -r 's#(.*):(.*)#\\1#' > ngrok_host.txt")
    get_ipython().system("cat ngrok_url.txt | sed 's|^tcp://||' | sed -r 's#(.*):(.*)#\\2#'  > ngrok_port.txt")

    setvar("""
    MACHINE_IP=$(cat ngrok_host.txt)
    PORT=$(cat ngrok_port.txt)
    """)

readpass("PBTOK")

readpass("AGAVE_PASSWD")

get_ipython().system('tenants-init -t sandbox')

get_ipython().system('clients-delete -u $AGAVE_USERNAME -p "$AGAVE_PASSWD" $APP_NAME')

get_ipython().system('clients-create -p "$AGAVE_PASSWD" -S -N $APP_NAME -u $AGAVE_USERNAME')

get_ipython().system('auth-tokens-create -u $AGAVE_USERNAME -p "$AGAVE_PASSWD"')



