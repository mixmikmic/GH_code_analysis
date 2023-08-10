import os
import sys
import time
sys.path.append("..")
get_ipython().magic('reload_ext autoreload')

directory = "../examplepackage/"
if not os.path.exists(directory):
    os.makedirs(directory)

get_ipython().run_cell_magic('writefile', '../examplepackage/neato.py', "\ndef torpedo():\n    print('First module modification 0!')")

get_ipython().run_cell_magic('writefile', '../examplepackage/neato2.py', "\ndef torpedo2():\n    print('Second module modification 0!')")

get_ipython().run_cell_magic('writefile', '../examplepackage/neato3.py', "\ndef torpedo3():\n    print('Third module modification 0!')")

# when hitting 'run all' this needs a short delay (probable race condition).
time.sleep(1.5)

import examplepackage.neato
import examplepackage.neato2
import examplepackage.neato3

get_ipython().magic('autoreload 1')
get_ipython().magic('aimport examplepackage')

examplepackage.neato.torpedo()

examplepackage.neato2.torpedo2()

examplepackage.neato3.torpedo3()

get_ipython().run_cell_magic('writefile', '../examplepackage/neato.py', "\ndef torpedo():\n    print('First module modification 1')")

get_ipython().run_cell_magic('writefile', '../examplepackage/neato2.py', "\ndef torpedo2():\n    print('Second module modification 1')")

get_ipython().run_cell_magic('writefile', '../examplepackage/neato3.py', "\ndef torpedo3():\n    print('Third module modification 1!')")

# when hitting 'run all' this needs a short delay (probable race condition).
time.sleep(1.5)

examplepackage.neato.torpedo()

examplepackage.neato2.torpedo2()

examplepackage.neato3.torpedo3()

get_ipython().magic('autoreload 1')
get_ipython().magic('aimport examplepackage.neato')

examplepackage.neato.torpedo()

examplepackage.neato2.torpedo2()

examplepackage.neato3.torpedo3()

get_ipython().magic('autoreload 2')
get_ipython().magic('aimport examplepackage.neato')
get_ipython().magic('aimport -examplepackage.neato2')

examplepackage.neato.torpedo()

examplepackage.neato2.torpedo2()

examplepackage.neato3.torpedo3()

get_ipython().run_cell_magic('writefile', '../examplepackage/neato.py', "\ndef torpedo():\n    print('First module modification 2!')")

get_ipython().run_cell_magic('writefile', '../examplepackage/neato2.py', "\ndef torpedo2():\n    print('Second module modification 2!')")

get_ipython().run_cell_magic('writefile', '../examplepackage/neato3.py', "\ndef torpedo3():\n    print('Third module modification 2!')")

# when hitting 'run all' this needs a short delay (race condition).
time.sleep(1.5)

examplepackage.neato.torpedo()

examplepackage.neato2.torpedo2()

examplepackage.neato3.torpedo3()



