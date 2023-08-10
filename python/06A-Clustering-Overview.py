get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")
# import libraries
import numpy as np
import matplotlib as mp
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import slideUtilities as sl
import laUtilities as ut
from importlib import reload
from datetime import datetime
from IPython.display import Image
from IPython.display import display_html
from IPython.display import display
from IPython.display import Math
from IPython.display import Latex
from IPython.display import HTML
print('')

get_ipython().run_cell_magic('html', '', '<style>\n .container.slides .celltoolbar, .container.slides .hide-in-slideshow {\n    display: None ! important;\n}\n</style>')

sl.hide_code_in_slideshow()
display(Image("figs/L6-Snow-cholera-map-1.png", width=550))

sl.hide_code_in_slideshow()
display(Image("figs/L6-John_Snow_memorial_and_pub.png", width=550))

# source: http://www.randomservices.org/random/data/HorseKicks.html
import pandas as pd
df = pd.read_table('data/HorseKicks.txt',index_col='Year',dtype='float')
counts = df.sum(axis=1)
counts

_ = counts.hist(bins=25,xlabelsize=16)

counts.mean()

from sklearn import preprocessing
counts_scaled = pd.DataFrame(preprocessing.scale(counts))
_ = counts_scaled.hist(bins=25,xlabelsize=16)

counts_scaled.mean().values

min_max_scaler = preprocessing.MinMaxScaler()
counts_minmax = min_max_scaler.fit_transform(counts.values.reshape(-1,1))
counts_minmax = pd.DataFrame(counts_minmax)
_ = counts_minmax.hist(bins=25,xlabelsize=16)

sl.hide_code_in_slideshow()
display(Image("figs/L6-annie19980405.jpg", width=350))

sl.hide_code_in_slideshow()
display(Image("figs/L6-annie_002.png", width=550))

sl.hide_code_in_slideshow()
display(Image("figs/L6-annie_004.png", width=550))  

sl.hide_code_in_slideshow()
display(Image("figs/L6-annie_008.png", width=550)) 

sl.hide_code_in_slideshow()
display(Image("figs/L6-annie_016.png", width=550)) 



