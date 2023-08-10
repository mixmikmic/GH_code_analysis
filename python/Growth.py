get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import ipystata

get_ipython().run_cell_magic('stata', '', '\nuse http://wps.pearsoned.co.uk/wps/media/objects/12401/12699039/empirical/empex_tb/Growth.dta, clear\nsum')

get_ipython().run_cell_magic('stata', '--graph', 'two (scatter growth tradeshare)')

get_ipython().run_cell_magic('stata', '--graph', '\ntwo (scatter growth tradeshare, mcolor(blue)) /// \n    (scatter growth tradeshare if country_name=="Malta", mcolor(red)) /// \n    , scheme(s1color) legend(pos(7) ring(0) label(1 "All countries") label(2 "Malta"))\n/*\nMalta does look as an outlier in the sense that its value of trade share is abnormaly distant from other values.\n*/')

get_ipython().run_cell_magic('stata', '--graph', '\nreg growth tradeshare, r \n\n\npredict growthhat\n\nreg growth tradeshare if (country_name!="Malta"), r \n\n\npredict growthhat_nomalta\n\n\ntwo (scatter growth tradeshare , mcolor(black)) /// \n    (line growthhat tradeshare , lwidth(medthick) lpattern(solid) lcolor(blue)) /// \n    (line growthhat_nomalta tradeshare , lwidth(medthick) lpattern(solid) lcolor(red)) /// \n    , scheme(s1color) legend(pos(7) ring(0)label(1 "Observations") label(2 "fitted values all") label(3 "fitted values w/o malta"))\n\n')

get_ipython().run_cell_magic('stata', '', '*//exclude data from Malta \ndrop if country_name=="Malta"')

get_ipython().run_cell_magic('stata', '', 'sum\nreg growth tradeshare yearsschool rev_coups assasinations rgdp60, r')

get_ipython().run_cell_magic('stata', '', 'test yearsschool rev_coups assasinations rgdp60')















