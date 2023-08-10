
get_ipython().system('pip install --upgrade numpy')
get_ipython().system('pip install --upgrade scipy')
get_ipython().system('pip install --upgrade statsmodels')


import numpy as np
from scipy.stats import sem, t
import statsmodels.stats.api as sms

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

a = range(10, 14)


t.interval(0.95, len(a)-1, loc=np.mean(a), scale=sem(a))


sms.DescrStatsW(a).tconfint_mean()


mean_confidence_interval(a)


get_ipython().run_line_magic('pprint', '')
from scipy import stats

[d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]


get_ipython().run_line_magic('pprint', '')
from scipy import stats

[d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_discrete)]



