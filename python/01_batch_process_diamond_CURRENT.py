import numpy as np
import pandas as pd
import os
from qtools import Submitter
import glob
from tqdm import tnrange, tqdm_notebook
from Bio import SeqIO

fasta = '/home/bay001/projects/kes_20160307/permanent_data/current/current_fasta/whole/kestrel5-reclustered.no-aerv.no-mtdna.fasta'
outdir = '/home/bay001/projects/kes_20160307/permanent_data/current/current_fasta/parts/'
handle = open(fasta, "rU")
entries = 0 # entries per part counter
counter = 1 # parts counter
records = []
MAX_ENTRIES = 200
output = os.path.join(outdir, 'part_{}.fas'.format(counter))

progress = tnrange(1425)
for record in SeqIO.parse(handle, "fasta"):
    records.append(record)
    
    if entries >= MAX_ENTRIES:
        counter += 1
        SeqIO.write(records, output, "fasta")
        output = os.path.join(outdir, 'part_{}.fas'.format(counter))
        entries = 0
        records = []
        progress.update(1)
    entries += 1

fasta_dir = '/home/bay001/projects/kes_20160307/permanent_data/current/current_fasta/parts'
db = '/home/bay001/projects/kes_20160307/data/uniref90.db.dmnd'
parts = glob.glob(os.path.join(fasta_dir, '*.fas'))
print("there are {} parts to diamond".format(len(parts)))
cmds = []
outdir = '/home/bay001/projects/kes_20160307/data/diamond/'
for part in sorted(parts):
    output = part + '.annotated.fasta'
    cmd = '/projects/ps-yeolab3/bay001/software/diamond blastx '
    cmd += '-d {} '.format(db)
    cmd += '-q {} '.format(part)
    cmd += '-o {} '.format(output)
    cmd += '--more-sensitive'
    cmds.append(cmd)

bash_script = '/home/bay001/projects/kes_20160307/permanent_data/current/current_fasta/bash_scripts/part1000_1250_.sh'

Submitter(
    cmds[1000:1250], job_name='PNG500', array=True, submit=False, 
    walltime='8:00:00', queue='condo', sh=bash_script, ppn=10, nodes=1
)

cmds



