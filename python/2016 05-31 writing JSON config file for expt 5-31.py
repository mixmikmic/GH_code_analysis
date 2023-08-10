import numpy as np
import json
from os.path import expanduser

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

whitenoise_rf_mapping = '{"function": "whitenoise", "length": 5, "seed": 150, "framerate": 30, "contrast": 1.0, "dist": "binary", "ndims": [50,50]}'
naturalscene_repeat = '{"function": "naturalscene", "length": 0.5, "seed": 90, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "imgdir": "images/", "imgext": "*.mat", "jumpevery": 30, "jitter": 0.5}'
naturalscene_clips = ['{"function": "naturalscene", "length": 5, "seed": %d, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "imgdir": "images/", "imgext": "*.mat", "jumpevery": 30, "jitter": 0.5}' %(91+i)
                      for i in range(14)]

whitenoise_rf_mapping

naturalscene_clips

stimulus_seq = []
stimulus_seq.append(whitenoise_rf_mapping)
# 14 single trials of 5 min each (70 min total)
for block in range(len(naturalscene_clips)):
    stimulus_seq.append(naturalscene_clips[block])
    # 14 * 8 = 112 repeats
    for repeat in range(8):
        stimulus_seq.append(naturalscene_repeat)
stimulus_seq.append(whitenoise_rf_mapping)

len(stimulus_seq)

total_length = 5*2 + 14.*8.*.5 + 14*5
total_length

with open('config.json', 'w') as outfile:
    json.dump(stimulus_seq, outfile)

stimulus_seq

whitenoise_rf_mapping = {"function": "whitenoise", "length": 5, "seed": 150, "framerate": 30, "contrast": 1.0, "dist": "binary", "ndims": [50,50]}
naturalscene_repeat = {"function": "naturalscene", "length": 0.5, "seed": 90, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "imgdir": "images/", "imgext": "*.mat", "jumpevery": 30, "jitter": 0.5}
naturalscene_clips = [{"function": "naturalscene", "length": 5, "seed": i+91, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "imgdir": "images/", "imgext": "*.mat", "jumpevery": 30, "jitter": 0.5} for i in range(14)]

stimulus_seq = []
stimulus_seq.append(whitenoise_rf_mapping)
# 14 single trials of 5 min each (70 min total)
for block in range(len(naturalscene_clips)):
    stimulus_seq.append(naturalscene_clips[block])
    # 14 * 8 = 112 repeats
    for repeat in range(8):
        stimulus_seq.append(naturalscene_repeat)
stimulus_seq.append(whitenoise_rf_mapping)

with open('config.json', 'w') as outfile:
    json.dump(stimulus_seq, outfile)



