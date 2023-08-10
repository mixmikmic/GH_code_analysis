__depends__=[]
__dest__="../results/sample_table.tex"

import numpy as np
import scipy.stats
import astropy.table
import astropy.io

sample_sizes = np.array([int(s) for s in np.logspace(1,4,4)])

true_mu,true_sigma = 1.,0.5
distributions = []
for s in sample_sizes:
    distributions.append(np.random.normal(loc=true_mu,scale=true_sigma,size=s))

fitted_mu,fitted_sigma = [],[]
for s,d in zip(sample_sizes,distributions):
    mu,sigma = scipy.stats.norm.fit(d)
    fitted_mu.append(mu)
    fitted_sigma.append(sigma)
fitted_mu = np.array(fitted_mu)
fitted_sigma = np.array(fitted_sigma)

mu_errors = np.fabs(fitted_mu - true_mu)/(fitted_mu + true_mu)*100.

headers = (r'$N$',r'$\mu$',r'$\sigma$',r'$\varepsilon_{\mu}$')
results_table = astropy.table.Table(np.vstack((sample_sizes,fitted_mu,fitted_sigma,mu_errors)).T,names=headers)

results_table

formats = {
    r'$N$':'%d',
    r'$\mu$':'.3f',r'$\sigma$':'.3f',r'$\varepsilon_{\mu}$':'.3f'
}
caption = r'''This table is generated on the fly in the AAS\TeX \texttt{deluxetable} style using AstroPy. It can be
easily viewed in the Jupyter notebook and is a great way for easily rebuilding tables each time your data changes.
'''

astropy.io.ascii.write(results_table, 
                       output=__dest__,
                       format='aastex',
                       formats=formats,
                       caption=caption
                      )



