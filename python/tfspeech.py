# # Shell script for preparing data
# cd data
# mkdir tfspeech
# cd tfspeech

# kg download -u <username> -p <password> -c tensorflow-speech-recognition-challenge -f sample_submission.7z
# kg download -u <username> -p <password> -c tensorflow-speech-recognition-challenge -f test.7z
# kg download -u <username> -p <password> -c tensorflow-speech-recognition-challenge -f train.7z

# 7z x test.7z
# 7z x train.7z
# 7z x sample_submission.7z







# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "data/tfspeech/"
sz = 224

