import pandas as pd
import numpy as np
import os
import glob
from qtools import Submitter

wd = '/projects/ps-yeolab3/bay001/maps/current_annotations/'
prog = '/home/bay001/projects/codebase/bfx/pyscripts/rnaseq/subset_rmats_junctioncountonly.py'
se_mats = glob.glob(os.path.join(wd,'*-SE.MATS.JunctionCountOnly.positive.txt'))
se_mats[:5]

for se in se_mats:
    positive_input = se
    negative_input = se.replace('.positive.txt','.negative.txt')
    
    positive_output = positive_input.replace('.txt','.nr.txt')
    negative_output = negative_input.replace('.txt','.nr.txt')
    
    get_nr_positive = "python {} -i {} -o {}".format(prog, positive_input, positive_output)
    get_nr_negative = "python {} -i {} -o {}".format(prog, negative_input, negative_output)
    cmdstr = [get_nr_positive, get_nr_negative]
    jobname = os.path.basename(se).split('-')[0]
    qtools.Submitter(
        cmdstr, 
        jobname, 
        array=False, 
        nodes=1, 
        ppn=1, 
        walltime='0:30:00', 
        submit=True, 
        queue='home-scrm',
        sh='/home/bay001/projects/codebase/temp/bash_scripts/{}.sh'.format(jobname)
    )



