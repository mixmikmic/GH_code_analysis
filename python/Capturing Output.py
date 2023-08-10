from __future__ import print_function
import sys

get_ipython().run_cell_magic('capture', '', "print('hi, stdout')\nprint('hi, stderr', file=sys.stderr)")

get_ipython().run_cell_magic('capture', 'captured', "print('hi, stdout')\nprint('hi, stderr', file=sys.stderr)")

captured

captured()

captured.stdout

captured.stderr

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_cell_magic('capture', 'wontshutup', '\nprint("setting up X")\nx = np.linspace(0,5,1000)\nprint("step 2: constructing y-data")\ny = np.sin(x)\nprint("step 3: display info about y")\nplt.plot(x,y)\nprint("okay, I\'m done now")')

wontshutup()

get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'print(\'hi, stdout\')\nprint("hello, stderr", file=sys.stderr)')

cap.stdout

cap.stderr

cap.outputs

