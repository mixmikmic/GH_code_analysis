import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmagpy.pmag as pmag
import pmagpy.ipmag as ipmag
import pmagpy.pmagplotlib as pmag_plot
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_formats = {'svg',}")

fisher_directions = ipmag.fishrot(k=40, n=20, dec=200, inc=50)
directions = pd.DataFrame(fisher_directions,columns=['dec','inc','length'])
directions.head()

fisher_mean = ipmag.fisher_mean(directions.dec,directions.inc)

plt.figure(num=1,figsize=(5,5))
ipmag.plot_net(1)
ipmag.plot_di(dec=directions.dec,inc=directions.inc)
ipmag.plot_di_mean(fisher_mean['dec'],fisher_mean['inc'],fisher_mean['alpha95'],
                   marker='s',color='r')

# squish all inclinations
squished_incs = []
for inclination in directions.inc:
    squished_incs.append(ipmag.squish(inclination, 0.4))

# plot the squished directional data
plt.figure(num=1,figsize=(5,5))
ipmag.plot_net(1)
ipmag.plot_di(directions.dec,squished_incs)
squished_DIs = np.array(zip(directions.dec,squished_incs))

bingham_mean = ipmag.bingham_mean(directions.dec,squished_incs)
bingham_mean

plt.figure(num=1,figsize=(5,5))
ipmag.plot_net(1)
ipmag.plot_di(directions.dec,squished_incs)
ipmag.plot_di_mean_bingham(bingham_mean,color='red',marker='s')



