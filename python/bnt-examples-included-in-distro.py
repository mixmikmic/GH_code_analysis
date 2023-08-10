import os
import sys
nb_dir = os.getcwd()
print(nb_dir)
os.chdir('../../')
qfog_path = os.getcwd()
print(qfog_path)
sys.path.insert(0,qfog_path)

bnt_path = '/home/jupyter/Notebooks/Classical/bnt'

import oct2py
get_ipython().magic('load_ext oct2py.ipython')

get_ipython().run_cell_magic('octave', '', "do_braindead_shortcircuit_evaluation (1)\nwarning('off', 'Octave:possible-matlab-short-circuit-operator')")

get_ipython().magic('octave_push bnt_path')
# genpath(dir) grows dir to list of all files inside it (recursive)
get_ipython().magic('octave addpath(genpath(bnt_path))')

fdir = bnt_path + '/BNT/examples/static/' 
get_ipython().magic('ls $fdir')

def do_file(fname):
    fpath =  fdir + fname
    print(fpath)
    with open(fpath) as f:
        print(f.read()) # prints whole file
    print('_________________________output______________________________')
    fname1 = fname[:-2] # removes '.m'  
    get_ipython().magic('octave $fname1')

do_file('burglary.m')

do_file('brainy.m')



