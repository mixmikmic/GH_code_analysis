get_ipython().magic('matplotlib inline')

import mpl_plot_templates

x = np.random.randn(10000)
y = np.random.randn(10000)

import pylab as pl

fig = pl.figure(1)
pl.subplot(1,2,1, rasterized=True)
pl.plot(x,y,'.')
pl.subplot(1,2,2)
rslt = mpl_plot_templates.adaptive_param_plot(x, y)
fig.savefig('rasterized.pdf')

fig = pl.figure(1)
pl.subplot(1,2,1, rasterized=False)
pl.plot(x,y,'.')
pl.subplot(1,2,2)
rslt = mpl_plot_templates.adaptive_param_plot(x, y)
fig.savefig('not_rasterized.pdf')

get_ipython().run_cell_magic('bash', '', '# non-rasterized files are somewhat larger\nls -lh *raster*\nopen *raster*pdf')



