get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import ipystata

get_ipython().run_cell_magic('stata', '', '\nuse http://wps.pearsoned.co.uk/wps/media/objects/12401/12699039/empirical/empex_tb/SeatBelts.dta, clear\nsum')

get_ipython().run_cell_magic('stata', '', '*gen logincome = log(income)\nreg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age, vce(cluster STATE) ')



get_ipython().run_cell_magic('stata', '', '*http://www.stata.com/support/faqs/data-management/creating-dummy-variables/\ntabulate state, generate(stateB)\n\n*Least squares with dummy variables (no constant term) .\n \nreg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age stateB*,  vce(cluster STATE) nocons')

get_ipython().run_cell_magic('stata', '', 'reg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age stateB*,  vce(cluster STATE) \ntestparm stateB*')

get_ipython().run_cell_magic('stata', '', 'reg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age i.STATE,  vce(cluster STATE) ')

get_ipython().run_cell_magic('stata', '', 'testparm STATE')







get_ipython().run_cell_magic('stata', '', '\n\n*http://www.stata.com/support/faqs/data-management/creating-group-identifiers/\n\n* egen STATE = group(state)\n\n*Within estimation\nxtreg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age, fe i(STATE) vce(cluster STATE)')

get_ipython().run_cell_magic('stata', '', '\n\n*http://www.stata.com/support/faqs/data-management/creating-group-identifiers/\n\n* egen STATE = group(state)\n\n*Within estimation\nxtreg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age i(year), fe i(STATE) r')



get_ipython().run_cell_magic('stata', '', '\n\n*http://www.stata.com/support/faqs/data-management/creating-group-identifiers/\n\n* egen STATE = group(state)\n\n*Within estimation\nxtreg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age, fe i(STATE) ')



get_ipython().run_cell_magic('stata', '', 'egen STATE = group(state)\nxtset STATE year')

get_ipython().run_cell_magic('stata', '', '*gen logincome = log(income)\nxtreg fatalityrate  sb_useage speed65 speed70 ba08 drinkage21 logincome age')



