get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import ipystata

get_ipython().run_cell_magic('stata', '', '\n*CIG85_95: N=528, Panel data, annual per capita cigarette sales for 48 states in packs per fiscal year from 1985-1995.\nuse http://fmwww.bc.edu/ec-p/data/stockwatson/cig85_95, clear\ngen lpackpc = log(packpc)\ngen lperinc = log(income)\n\nregress lpackpc tax lperinc\n')

get_ipython().run_cell_magic('stata', '', '*Before-After estimation\ngen diff_rtax= rtax1995- rtax1985 \ngen diff_lpackpc= lpackpc1995- lpackpc1985 \ngen diff_lperinc= lperinc1995- lperinc1985\n\nregress diff_lpackpc diff_rtax diff_lperinc, nocons\n\n')

get_ipython().run_cell_magic('stata', '', '*Before-After estimation\n*gen diff_rtax= rtax1995- rtax1985 \n*gen diff_lpackpc= lpackpc1995- lpackpc1985 \n*gen diff_lperinc= lperinc1995- lperinc1985\n\n*regress diff_lpackpc diff_rtax diff_lperinc, nocons\n\n\n*http://www.stata.com/support/faqs/data-management/creating-dummy-variables/\ntabulate state, generate(stateB)\n\n*Least squares with dummy variables (no constant term) .\n \nregress lpackpc tax lperinc stateB*, nocons\n')

get_ipython().run_cell_magic('stata', '', '\n*Least squares with dummy variables with constant term . \nregress lpackpc tax lperinc stateB*\n\n\n')

get_ipython().run_cell_magic('stata', '', '\n\n*http://www.stata.com/support/faqs/data-management/creating-group-identifiers/\n\negen STATE = group(state)\n\n*Within estimation\nxtreg lpackpc tax lperinc, fe i(STATE)\n')



