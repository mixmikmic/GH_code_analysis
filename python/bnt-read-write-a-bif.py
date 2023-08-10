import os
import sys
nb_dir = os.getcwd()
print(nb_dir)
os.chdir('../../')
qfog_path = os.getcwd()
print(qfog_path)
sys.path.insert(0,qfog_path)

bnt_path = '/home/jupyter/Notebooks/Classical/bnt'

import numpy as np
import operator
import oct2py
get_ipython().magic('load_ext oct2py.ipython')

from graphs.BayesNet import *

get_ipython().run_cell_magic('octave', '', "do_braindead_shortcircuit_evaluation (1)\nwarning('off', 'Octave:possible-matlab-short-circuit-operator')")

get_ipython().magic('octave_push bnt_path')
# genpath(dir) grows dir to list of all files inside it (recursive)
get_ipython().magic('octave addpath(genpath(bnt_path))')

# first a test that illustrates use of "global" keyword
def fun():
    global x100, y100
    x100=5
    y100=6
    # '%octave_push' gets confused if use commas between 
    # variables but 'global' requires them
    get_ipython().magic('octave_push x100 y100')
    
fun() # this gives error if do not declare x100 global
get_ipython().magic('octave disp(x100)')

get_ipython().magic('run jupyter-notebooks/bnt-examples/bnt-biftool.ipynb')

def test(prefix, verbose):
    vtx_to_states = bnt_read_bif(prefix + ".bif", verbose=verbose)
    # print(vtx_to_states)
    bnt_write_bif(prefix + "_bnt.bif", vtx_to_states=vtx_to_states, verbose=verbose)

    bnt_read_dot(prefix + ".dot", verbose=verbose)
    bnt_write_dot(prefix + "_bnt.dot", verbose=verbose)

    from graphviz import Digraph, Source
    from IPython.display import display
    display(Source(open(prefix + ".dot").read()))
    print("_________________________________________________________")
    display(Source(open(prefix + "_bnt.dot").read()))

prefix = "examples_cbnets/WetGrass"
test(prefix, True)

prefix = "examples_cbnets/asia"
test(prefix, False)

prefix = "examples_cbnets/earthquake"
test(prefix, False)



