get_ipython().system('mkdir -p ~/agave/funwave-tvd-docker')

get_ipython().run_line_magic('cd', '~/agave')

get_ipython().system('pip3 install setvar')

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
loadvar()
get_ipython().system('auth-tokens-refresh')

writefile("funwave-tvd-docker/Dockerfile","""
FROM stevenrbrandt/science-base
USER root
RUN mkdir -p /home/install
RUN mkdir -p /scratch
RUN chown jovyan /home/install
USER jovyan

MAINTAINER Steven R. Brandt <sbrandt@cct.lsu.edu>
RUN cd /home/install && \
    git clone https://github.com/fengyanshi/FUNWAVE-TVD && \
    cd FUNWAVE-TVD/src && \
    perl -p -i -e 's/FLAG_8 = -DCOUPLING/#$&/' Makefile && \
    make

WORKDIR /home/install/FUNWAVE-TVD/src
RUN mkdir -p /home/jovyan/rundir
WORKDIR /home/jovyan/rundir
""")

get_ipython().system('tar -czf dockerjob.tgz -C funwave-tvd-docker Dockerfile')
get_ipython().system('files-mkdir -S ${AGAVE_STORAGE_SYSTEM_ID} -N funwave-tvd-docker')
get_ipython().system('files-upload -F dockerjob.tgz -S ${AGAVE_STORAGE_SYSTEM_ID} funwave-tvd-docker/')

import runagavecmd as r
import imp
imp.reload(r)

r.runagavecmd(
    "tar xzf dockerjob.tgz && sudo docker build --rm -t funwave-tvd-2 .",
    "agave://${AGAVE_STORAGE_SYSTEM_ID}/funwave-tvd-docker/dockerjob.tgz"
)

get_ipython().system('jobs-output-get ${JOB_ID} fork-command-1.err')
get_ipython().system('cat fork-command-1.err')

writefile("rundock.sh","""
rm -fr cid.txt out.tgz

# Start a docker image running in detached mode, write the container id to cid.txt
sudo docker run -d -it --rm --cidfile cid.txt funwave-tvd-2 bash

# Store the container id in CID for convenience
CID=\$(cat cid.txt)

# Copy the input.txt file into the running image
sudo docker cp input.txt \$CID:/home/jovyan/rundir/

# Run funwave on the image
sudo docker exec --user jovyan \$CID mpirun -np 2 /home/install/FUNWAVE-TVD/src/funwave_vessel

# Extract the output files from the running image
# Having them in a tgz makes it more convenient to fetch them with jobs-output-get
sudo docker exec --user jovyan \$CID tar czf - output > out.tgz

# Stop the image
sudo docker stop \$CID

# List the output files
tar tzf out.tgz
""")

get_ipython().system('tar czf rundock.tgz rundock.sh input.txt')
get_ipython().system('files-upload -F rundock.tgz -S ${AGAVE_STORAGE_SYSTEM_ID} funwave-tvd-docker/')

r.runagavecmd(
    "tar xzf rundock.tgz && bash rundock.sh",
    "agave://${AGAVE_STORAGE_SYSTEM_ID}/funwave-tvd-docker/rundock.tgz")

get_ipython().system('jobs-output-list ${JOB_ID}')
get_ipython().system('jobs-output-get ${JOB_ID} out.tgz')
get_ipython().system('tar xzf out.tgz')

get_ipython().system('head output/eta_00010')

get_ipython().system('files-mkdir -S ${AGAVE_STORAGE_SYSTEM_ID} -N sing')
get_ipython().system('files-upload -F input.txt -S ${AGAVE_STORAGE_SYSTEM_ID} sing/')
r.runagavecmd(
            "mkdir -p ~/singu && "+
            "cd ~/singu && "+
            "rm -f funwave-tvd.img && "+
            "singularity create funwave-tvd.img --size 2000 && "+
            "singularity import funwave-tvd.img docker://stevenrbrandt/funwave-tvd-2:latest")

get_ipython().system('files-upload -F input.txt -S ${AGAVE_STORAGE_SYSTEM_ID} ./')
r.runagavecmd(
    "export LD_LIBRARY_PATH=/usr/local/lib && "+
    "mpirun -np 2 singularity exec ~/singu/funwave-tvd.img /home/install/FUNWAVE-TVD/src/funwave_vessel && "+
    "tar cvzf singout.tgz output",
    "agave://${AGAVE_STORAGE_SYSTEM_ID}/input.txt"
)

get_ipython().system('jobs-output-get ${JOB_ID} singout.tgz')
get_ipython().system('rm -fr output')
get_ipython().system('tar xzf singout.tgz')

get_ipython().system('head output/v_00003')

get_ipython().system('echo /home/jovyan/singu/funwave-tvd.img > ~/work/sing.txt')

get_ipython().system('files-upload -F input.txt -S ${AGAVE_STORAGE_SYSTEM_ID} ./')
r.runagavecmd(
    "rm -fr output && "+
    "mpirun -np 2 /home/install/FUNWAVE-TVD/src/funwave_vessel && "+
    "tar cvzf singout.tgz output",
    "agave://${AGAVE_STORAGE_SYSTEM_ID}/input.txt"
)

get_ipython().system('rm -fr output singout.tgz')
get_ipython().system('jobs-output-get ${JOB_ID} singout.tgz')
get_ipython().system('tar xzf singout.tgz')
get_ipython().system('ls output')

# Clean up so that we don't boot into the singularity image without intending to
get_ipython().system('rm -f ~/work/sing.txt')



