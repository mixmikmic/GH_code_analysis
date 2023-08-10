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

naturalscene_rf_mapping = '{"function": "naturalscene", "length": 5, "seed": 150, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "imgdir": "images/", "imgext": "*.mat", "jumpevery": 30, "jitter": 0.5}'
whitenoise_repeat = '{"function": "whitenoise", "length": 0.5, "seed": 90, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "dist": "binary"}'
whitenoise_clips = ['{"function": "whitenoise", "length": 5, "seed": %d, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "dist": "binary"}' %(91+i)
                      for i in range(14)]

naturalscene_rf_mapping

whitenoise_clips

stimulus_seq = []
stimulus_seq.append(naturalscene_rf_mapping)
# 14 single trials of 5 min each (70 min total)
for block in range(len(whitenoise_clips)):
    stimulus_seq.append(whitenoise_clips[block])
    # 14 * 8 = 112 repeats
    for repeat in range(8):
        stimulus_seq.append(whitenoise_repeat)
stimulus_seq.append(naturalscene_rf_mapping)

len(stimulus_seq)

total_length = 5*2 + 14.*8.*.5 + 14*5
total_length

with open('config.json', 'w') as outfile:
    json.dump(stimulus_seq, outfile)

stimulus_seq

naturalscene_rf_mapping = {"function": "naturalscene", "length": 5, "seed": 150, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "imgdir": "images/", "imgext": "*.mat", "jumpevery": 30, "jitter": 0.5}
whitenoise_repeat = {"function": "whitenoise", "length": 0.5, "seed": 90, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "dist": "binary"}
whitenoise_clips = [{"function": "whitenoise", "length": 5, "seed": i+91, "framerate": 30, "contrast": 1.0, "ndims": [50,50], "dist": "binary"} for i in range(14)]

stimulus_seq = []
stimulus_seq.append(naturalscene_rf_mapping)
# 14 single trials of 5 min each (70 min total)
for block in range(len(whitenoise_clips)):
    stimulus_seq.append(whitenoise_clips[block])
    # 14 * 8 = 112 repeats
    for repeat in range(8):
        stimulus_seq.append(whitenoise_repeat)
stimulus_seq.append(naturalscene_rf_mapping)

with open('config.json', 'w') as outfile:
    json.dump(stimulus_seq, outfile)





