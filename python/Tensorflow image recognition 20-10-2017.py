import sys
import subprocess
print(sys.version)
import tensorflow as tf

get_ipython().run_cell_magic('time', '', "py_to_run = '/Users/robincole/Documents/Github/Tensorflow/models/tutorials/image/imagenet/classify_image.py'\nprint(subprocess.check_output(['python3', py_to_run]))")



