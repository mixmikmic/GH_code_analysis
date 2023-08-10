get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import ipystata

get_ipython().run_cell_magic('stata', '', 'clear all\nuse http://fmwww.bc.edu/ec-p/data/stockwatson/fatality')

get_ipython().run_cell_magic('stata', '', 'sum')

get_ipython().run_cell_magic('stata', '--graph', 'twoway (scatter mrall beertax)')

get_ipython().run_cell_magic('stata', '--graph', 'twoway (scatter mrall beertax) (lfit mrall beertax)')

get_ipython().run_cell_magic('stata', '', 'gen frate = mrall*1000\nregress frate beertax if(year==1982), robust')

get_ipython().run_cell_magic('stata', '', 'sum')

