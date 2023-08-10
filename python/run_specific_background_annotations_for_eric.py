import pandas as pd
import numpy as np
import os
import glob

ctrls = glob.glob('/home/bay001/projects/maps_20160420/analysis/tests/eric_new_ctrls/*.miso')
ctrls = sorted(ctrls)
ctrls

script = '/home/bay001/projects/codebase/rbp-maps/maps/plot_peak_ericbg.py'
bedfile = '/home/bay001/projects/maps_20160420/analysis/tests/eric_new_ctrls/272.01v02.IDR.out.0102merged.bed'
cmd = 'python {} '.format(script)
cmd = cmd + '-i {} '.format(bedfile)
cmd = cmd + '-o {} '.format(bedfile + '.png')
cmd = cmd + '-m '
for c in ctrls:
    cmd = cmd + '{} '.format(c)
cmd

eric = pd.read_table('/home/bay001/projects/maps_20160420/analysis/tests/eric_new_ctrls/test.bed', index_col=0, names=['a'])
brian = pd.read_table('/home/bay001/projects/maps_20160420/analysis/tests/eric_new_ctrls/272.01v02.IDR.out.0102merged.bed.hepg2-nse-all.txt', names=['b'])

# 
pd.concat([eric,brian], axis=1)



