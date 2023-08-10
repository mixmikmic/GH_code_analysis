
get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import ipystata



get_ipython().run_cell_magic('stata', '--graph', 'clear all\nset seed 2016\nlocal T = 200\nset obs `T\'\ngen time = _n\nlabel var time "Time"\ntsset time\ngen eps = rnormal(0,5)\n\n\ngen yrw = eps in 1\nreplace yrw = l.yrw + eps in 2/l\n\n\ngen yrwd1 = 0.1 + eps in 1\nreplace yrwd1 = 0.1 + l.yrwd1 + eps in 2/l\n\n\n\ntsline yrw yrwd1, title("Stochastic trend")          ///\n        legend(label(1 "Random walk")                ///\n        label(2 "Random walk with drift"))\n\n    ')



get_ipython().run_cell_magic('stata', '--graph', '\n\n\ngen yrwd2 = 1 + eps in 1\nreplace yrwd2 = 1 + l.yrwd2 + eps in 2/l\n\n\ngen yt = 0.5 + 0.1*time + eps in 1\nreplace yt = 0.5 + 0.1*time +0.8*l.yt+ eps in 2/l\ndrop in 1/50\n\ntsline yt yrwd2,                                     ///\n        legend(label(1 "Deterministic time trend")   ///\n        label(2 "Random walk with drift"))           ///\n        title("Stochastic and deterministic trend")')



get_ipython().run_cell_magic('stata', '', '*gen yrwd2 = 1 + eps in 1\n*replace yrwd2 = 1 + l.yrwd2 + eps in 2/l\n*drop in 1/50\ndfuller yrwd2, trend')

get_ipython().run_cell_magic('stata', '', '\n\n*gen yt = 0.5 + 0.1*time + eps in 1\n*replace yt = 0.5 + 0.1*time +0.8*l.yt+ eps in 2/l\n\ndfuller yt, trend')



get_ipython().run_cell_magic('stata', '', '*gen yrwd2 = 1 + eps in 1\n*replace yrwd2 = 1 + l.yrwd2 + eps in 2/l\ndfgls yrwd2, maxlag(4)')

get_ipython().run_cell_magic('stata', '', 'dfgls yt, maxlag(4)')

get_ipython().run_cell_magic('stata', '', 'var yt, lags(1/2) nocons dfk small')

get_ipython().run_cell_magic('stata', '', 'varstable')

get_ipython().run_cell_magic('stata', '--graph', '\nvarbasic yt')

get_ipython().run_cell_magic('stata', '--graph', 'irf create ar2, set(myirf, replace)')

get_ipython().run_cell_magic('stata', '--graph', 'irf graph irf \nirf graph cirf')











get_ipython().run_cell_magic('stata', '--graph', 'clear all\nset seed 2016\nlocal T = 200\nset obs `T\'\ngen time = _n\nlabel var time "Time"\ntsset time\ngen eps = rnormal(0,5)\n\n\ngen yrw = eps in 1\nreplace yrw = l.yrw + eps in 2/l\n\n\ngen yrwd1 = 0.1 + eps in 1\nreplace yrwd1 = 0.1 + l.yrwd1 + eps in 2/l')



get_ipython().run_cell_magic('stata', '', 'gen yrwd2 = 1 + eps in 1\nreplace yrwd2 = 1 + l.yrwd2 + eps in 2/l\n\n\ngen yt = 0.5 + 0.1*time + eps in 1\nreplace yt = 0.5 + 0.1*time +0.8*l.yt+ eps in 2/l\ndrop in 1/50')









